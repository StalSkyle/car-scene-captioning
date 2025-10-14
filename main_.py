from PIL import Image
from io import BytesIO
import requests
import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import Optional


class ImageToText:
    """
    Модель, которая по фотографии машины
    автоматически генерирует краткое текстовое описание окружающей среды.

    Модель сначала прогоняет картинку через предобработку
    (коррекция ориентации и определение автомобиля),
    затем выделяет признаки с помощью вспомогательных моделей,
    и, наконец, получает текстовое описание от мультимодальной модели.
    """

    def __init__(self, path: str, source: bool = True, device: Optional[str] = None):
        """
        :param path: путь к фотографии (локальный или URL)
        :param source: True если фотография лежит локально, False если задан URL
        :param device: 'cuda' или 'cpu'. Если None, выбирается автоматически.
        """
        self.path = path
        self.source = source
        self.image = None

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rotation_model = None

        self.__init_rotate_model()

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

        self.rotation_model = rotation_model

    def change_path(self, path: str) -> None:
        """Меняет путь к изображению."""
        self.path = path

    def change_mode(self, source: bool) -> None:
        """Меняет режим работы загрузчика изображений."""
        self.source = source

    def load_image(self) -> bool:
        """
        Загружает изображение в переменную image в зависимости от источника фото.
        :return: True если успешно загружено, иначе False.
        """
        try:
            if self.source:
                self.image = Image.open(self.path).convert("RGB")
            else:
                response = requests.get(self.path, timeout=10)
                response.raise_for_status()
                self.image = Image.open(BytesIO(response.content)).convert("RGB")
            return True
        except Exception as e:
            print(f"Ошибка при загрузке изображения {self.path}: {e}")
            return False

    def show_loaded_image(self) -> None:
        """Показывает загруженное изображение."""
        if self.image is None:
            print('Изображение не загружено')
        else:
            self.image.show()

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

if __name__ == "__main__":
    # тут пока тесты

    # тесты на загрузчик
    itt = ImageToText('../../random.png')
    itt.show_loaded_image()

    itt.load_image()
    itt.show_loaded_image()

    itt.change_mode(False)
    itt.change_path('https://carsharing-acceptances.s3.yandex.net/000080b5-5f40-4f6b-56bd-68e42cfe0f1d/car_location_22075860-8af6-11f0-b981-052fecb11955.CAP84636423206376342.jpg/46f3928-fad162a0-af3ae0d5-516d116e')
    itt.load_image()
    itt.show_loaded_image()

    itt.rotate_image()
    itt.show_loaded_image()

