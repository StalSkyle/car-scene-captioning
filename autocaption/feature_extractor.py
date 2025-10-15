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
