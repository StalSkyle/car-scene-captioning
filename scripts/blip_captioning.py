import json
import os
import numpy as np
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    DetrImageProcessor, DetrForObjectDetection
)

# === Устройство ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Устройство: {device}")

# === Загрузка моделей ===
print("Загружаем BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

print("Загружаем DETR для детекции машины...")
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

# COCO class names, которые считаем "машиной"
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "train"}

def crop_to_environment(image_path):
    """
    Обрезает изображение так, чтобы оставить ТОЛЬКО окружение:
    - Находит bounding box машины
    - Берёт область НИЖЕ машины (где земля, здания, фонари)
    - Если машина не найдена — берёт нижнюю половину (без неба)
    """
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Детекция объектов
    inputs = detr_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detr_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = detr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]

    # Собираем bounding boxes для транспортных средств
    vehicle_boxes = []
    for label, box in zip(results["labels"], results["boxes"]):
        class_name = detr_model.config.id2label[label.item()]
        if class_name in VEHICLE_CLASSES:
            vehicle_boxes.append(box.cpu().numpy())

    if not vehicle_boxes:
        # Машина не найдена → берём нижнюю половину (без неба)
        return Image.fromarray(img_array[h//2:, :])

    # Находим самую нижнюю точку всех машин (ymax)
    max_ymax = max(box[3] for box in vehicle_boxes)
    crop_y1 = int(max_ymax)

    # Убедимся, что не вышли за границы
    if crop_y1 >= h:
        crop_y1 = h // 2

    cropped = img_array[crop_y1:, :]
    if cropped.size == 0:
        cropped = img_array[h//2:, :]

    return Image.fromarray(cropped)


def generate_car_environment_description(image_path):
    try:
        # Обрезаем до окружения
        env_image = crop_to_environment(image_path)

        # Генерация описания
        inputs = blip_processor(env_image, return_tensors="pt").to(device)

        with torch.no_grad():
            # Базовое
            outputs = blip_model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
            base = blip_processor.decode(outputs[0], skip_special_tokens=True)

            # Детальное
            detailed_outputs = blip_model.generate(**inputs, max_length=100, num_beams=7, length_penalty=2.0, early_stopping=True)
            detailed = blip_processor.decode(detailed_outputs[0], skip_special_tokens=True)

            # Альтернативное
            alt_outputs = blip_model.generate(**inputs, max_length=60, do_sample=True, temperature=0.9, top_p=0.9)
            alt = blip_processor.decode(alt_outputs[0], skip_special_tokens=True)

        return {
            "base": base,
            "detailed": detailed,
            "alternative": alt
        }

    except Exception as e:
        print(f"❌ Ошибка на {image_path}: {e}")
        # Fallback: использовать оригинальное изображение
        try:
            raw = Image.open(image_path).convert('RGB')
            inputs = blip_processor(raw, return_tensors="pt").to(device)
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_length=50, num_beams=5)
                desc = blip_processor.decode(out[0], skip_special_tokens=True)
            return {"base": desc, "detailed": desc, "alternative": desc}
        except:
            return {"base": "Ошибка", "detailed": "Ошибка", "alternative": "Ошибка"}


def process_images_with_blip(folder_path, output_json_path):
    results = {}
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(tuple(image_extensions))
    ]
    print(f"Найдено изображений: {len(image_files)}")

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(folder_path, img_file)
        print(f"[{i+1}/{len(image_files)}] Обработка: {img_file}")

        desc = generate_car_environment_description(img_path)
        results[img_file] = desc

        print(f"  Base: {desc['base']}")
        print(f"  Detailed: {desc['detailed']}")
        print()

    # Сохранение
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"✅ Результаты сохранены в: {output_json_path}")
    return results


# === ЗАПУСК ===
image_folder = "/home/jupyter/project/Grisha/dataset_images"
output_file = "/home/jupyter/project/Grisha/blip_environment_only.json"

results = process_images_with_blip(image_folder, output_file)