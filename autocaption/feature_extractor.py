import numpy as np
from PIL import Image
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

import torch
from torchvision import models, transforms
from ultralytics import YOLO


class ObjectExtractor:
    def __init__(self):

        self.image = None
        self.object_detection_model = self.__init_object_detection_model()

    @staticmethod
    def __init_object_detection_model():
        """Инициализирует модель для нахождения объектов на фотографии."""

        return YOLO('MODELS/yolov8x-oiv7.pt')

    def extract_features(self, image) -> dict[str, int]:
        results = self.object_detection_model(image, verbose=False)
        res_obj_detection = dict()
        detected = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = r.names[cls_id]
                confidence = float(box.conf)
                detected.append((label, confidence))

        for elem in detected:
            res_obj_detection[elem[0]] = elem[1]

        return res_obj_detection


class SceneExtractor:
    def __init__(self):
        self.scene_classification_model = self.__init_scene_classification_model()

    @staticmethod
    def __init_scene_classification_model():
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 7)

        checkpoint = torch.load('MODELS/resnet18_4x_scene_tag_bdd100k.pth',
                                map_location='cpu', weights_only=True)
        state_dict = checkpoint['state_dict']

        # убираем префиксы 'backbone.' и заменяем 'head.fc' на 'fc'
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_key = k[len('backbone.'):]
                new_state_dict[new_key] = v
            elif k == 'head.fc.weight':
                new_state_dict['fc.weight'] = v
            elif k == 'head.fc.bias':
                new_state_dict['fc.bias'] = v

        # загружаем
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        return model

    def predict_scene(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        x = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.scene_classification_model(x)
            probs = torch.sigmoid(logits).squeeze()  # multi-label

        labels = [
            "other",
            "highway",
            "residential",
            "city street",
            "parking lot",
            "gas stations",
            "tunnel"
        ]
        return {lbl: float(p) for lbl, p in zip(labels, probs)}


class PhotoDescriber:
    def __init__(self):
        self.CAR_CLASS_ID = 3  # COCO class ID for "car"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = self.initialize_models()

    def initialize_models(self):
        """
        Загружает и возвращает все необходимые модели и процессоры.
        """
        print("🔄 Инициализация моделей...")

        # BLIP для генерации описаний
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

        # Mask2Former для сегментации
        seg_processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        seg_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic").to(self.device)

        models = {
            "blip_processor": blip_processor,
            "blip_model": blip_model,
            "seg_processor": seg_processor,
            "seg_model": seg_model,
        }

        print("✅ Модели успешно инициализированы!")
        return models

    def remove_car_and_sky(self, image):
        """
        Возвращает изображение без машины и неба (через inpainting)
        """
        img = np.array(image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        seg_processor = self.models["seg_processor"]
        seg_model = self.models["seg_model"]

        inputs = seg_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = seg_model(**inputs)
        result = seg_processor.post_process_panoptic_segmentation(outputs, target_sizes=[img.shape[:2]])[0]

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if "masks" in result and "segments_info" in result:
            for seg_info in result["segments_info"]:
                if seg_info["id"] in result["masks"]:
                    if seg_info["label_id"] == self.CAR_CLASS_ID:
                        car_mask = (result["masks"] == seg_info["id"]).cpu().numpy().astype(np.uint8)
                        mask = np.maximum(mask, car_mask)

        # === Эвристика для неба ===
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([140, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        h, w = img.shape[:2]
        sky_candidate = np.zeros((h, w), dtype=np.uint8)
        sky_candidate[:h // 3, :] = 255

        sky_mask = cv2.bitwise_and(blue_mask, sky_candidate)
        sky_mask = (sky_mask > 0).astype(np.uint8)

        combined_mask = np.maximum(mask, sky_mask).astype(np.uint8)

        if combined_mask.sum() == 0:
            return image

        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)

        inpainted_bgr = cv2.inpaint(img_bgr, dilated_mask * 255, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(inpainted_rgb)

    def simple_photo_describe(self, image):
        """
        Обрабатывает одно изображение:
        - удаляет машину и небо,
        - генерирует 3 варианта описания окружения.

        Args:
            image (str): Изображение.

        Returns:
            dict: Словарь с ключами 'base', 'detailed', 'alternative'.
        """
        blip_processor = self.models["blip_processor"]
        blip_model = self.models["blip_model"]

        try:
            env_image = self.remove_car_and_sky(image)

            inputs = blip_processor(env_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Базовое описание
                outputs = blip_model.generate(**inputs, max_length=50, num_beams=5)
                base = blip_processor.decode(outputs[0], skip_special_tokens=True)

                # Подробное описание
                detailed_outputs = blip_model.generate(**inputs, max_length=100, num_beams=7, length_penalty=2.0)
                detailed = blip_processor.decode(detailed_outputs[0], skip_special_tokens=True)

                # Альтернативное (сэмплированное) описание
                alt_outputs = blip_model.generate(**inputs, max_length=60, do_sample=True, temperature=0.9, top_p=0.9)
                alt = blip_processor.decode(alt_outputs[0], skip_special_tokens=True)

            return {"base": base, "detailed": detailed, "alternative": alt}

        except Exception as e:
            print(f"⚠️ Ошибка при обработке {image}: {e}")
            # Fallback: используем оригинальное изображение
            raw_image = image
            inputs = blip_processor(raw_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_length=50, num_beams=5)
                desc = blip_processor.decode(out[0], skip_special_tokens=True)
            return {"base": desc, "detailed": desc, "alternative": desc}
