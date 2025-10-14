import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

from autocaption import ImageLoader, ImageProcessor, ObjectExtractor, SceneExtractor


def init_rotate_model(device):
    """Инициализирует модель для поворота изображения."""
    model_path = "MODELS/resnet50_rotation_car_99.76.pth"
    num_classes = 4

    rotation_model = models.resnet50(weights=None)
    num_features = rotation_model.fc.in_features
    rotation_model.fc = nn.Linear(num_features, num_classes)

    rotation_model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    rotation_model = rotation_model.to(device)
    rotation_model.eval()

    return rotation_model


def init_object_detection_model():
    """Инициализирует модель для нахождения объектов на фотографии."""

    return YOLO('MODELS/yolov8x-oiv7.pt')

def init_scene_classification_model():
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


def run_pipeline(image_path: list[str], source: bool) -> list:
    """
    Прогоняет картинку через модели с помощью написанной библиотеки autocaption
    :param image_path: путь к изображению
    :param source: источник картинки (True = локальный путь, False = URL)
    :return: текстовое описание окружения картинки
    """
    res = [[] * len(image_path)]

    # инициализация моделей
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rotation_model = init_rotate_model(device)
    object_detection_model = init_object_detection_model()
    scene_classification_model = init_scene_classification_model()

    # поехали
    for i, path in enumerate(image_path):
        image = ImageLoader(path=path, source=source).load_image()

        if image is None:
            res[i] = f"Не удалось загрузить изображение: {path}."
            continue

        res[i] = dict()

        # обработка изображения
        processor = ImageProcessor(rotation_model, image)
        processor.image = image
        processor.rotate_image()
        image = processor.image

        # нахождение признаков

        # нахождение предметов на фотографии
        object_extractor = ObjectExtractor(object_detection_model, image)
        res[i]['детекция объектов'] = object_extractor.extract_features()

        # определение сцены
        scene_extractor = SceneExtractor(scene_classification_model, image)
        res[i]['сцена'] = scene_extractor.predict_scene()

    return res


if __name__ == "__main__":
    # тесты
    print(run_pipeline(['../../parking.png'], source=True))
    # print(run_pipeline(['https://carsharing-acceptances.s3.yandex.net/000080b5-5f40-4f6b-56bd-68e42cfe0f1d/car_location_22075860-8af6-11f0-b981-052fecb11955.CAP84636423206376342.jpg/46f3928-fad162a0-af3ae0d5-516d116e'], source=False))
