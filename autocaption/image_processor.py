import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


class ImageRotator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rotation_model = self.__init_rotate_model()

    def __init_rotate_model(self):
        """Инициализирует модель для поворота изображения."""
        model_path = "MODELS/resnet50_rotation_car_99.76.pth"
        num_classes = 4

        rotation_model = models.resnet50(weights=None)
        num_features = rotation_model.fc.in_features
        rotation_model.fc = nn.Linear(num_features, num_classes)

        rotation_model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        rotation_model = rotation_model.to(self.device)
        rotation_model.eval()

        return rotation_model

    def rotate_image(self, image):
        """Вращает изображение, используя предобученную модель."""
        if image is None:
            raise "Изображение не загружено."

        angle_values = {0: 0, 1: 90, 2: 180, 3: 270}

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.rotation_model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        rotation_angle = angle_values[predicted_class]
        rotated_image = image.rotate(-rotation_angle, expand=True)

        return rotated_image

class CarDetector:
    def __init__(self):
        self.car_detection_model = self.__init_car_detection_model()

    @staticmethod
    def __init_car_detection_model():
        return YOLO('MODELS/car_detector.pt')

    def detect_car(self, image):

        result = self.car_detection_model(image, verbose=False)

        if result[0].boxes is None:
            return False

        for box in result[0].boxes:
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()

            class_name = self.car_detection_model.names[class_id]

            if class_name in ['car', 'truck'] and confidence > 0.25:
                return True

        return False
