import torch
import torch.nn as nn
from torchvision import models, transforms

class ImageProcessor:
    def __init__(self):
        self.rotation_model = None
        self.image = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def rotate_image(self) -> None:
        """Вращает изображение, используя предобученную модель."""
        if self.image is None:
            print("Изображение не загружено. Сначала вызовите load_image().")
            return

        angle_values = {0: 0, 1: 90, 2: 180, 3: 270}

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(self.image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.rotation_model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        rotation_angle = angle_values[predicted_class]
        rotated_image = self.image.rotate(-rotation_angle, expand=True)

        self.image = rotated_image
