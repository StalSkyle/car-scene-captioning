import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ObjectExtractor:
    def __init__(self, object_detection_model, image):
        self.object_detection_model = object_detection_model
        self.image = image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_features(self) -> list[str]:
        results = self.object_detection_model(self.image, verbose=False)
        res_obj_detection = []
        detected = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = r.names[cls_id]
                confidence = float(box.conf)
                detected.append((label, confidence))

        for elem in detected:
            res_obj_detection.append(elem)

        return res_obj_detection

class SceneExtractor:
    def __init__(self, scene_classification_model, image):
        self.scene_classification_model = scene_classification_model
        self.image = image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict_scene(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        x = transform(self.image).unsqueeze(0)

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