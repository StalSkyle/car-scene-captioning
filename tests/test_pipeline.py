import sys
import types

from conftest import load_source_module


def load_main_with_fakes(monkeypatch, *, loaded_image, has_car=True):
    counters = {
        "ImageRotator": 0,
        "CarDetector": 0,
        "ObjectExtractor": 0,
        "SceneExtractor": 0,
        "PhotoDescriber": 0,
        "PhotoDescriberWithQuestion": 0,
        "ImageLoader": 0,
        "rotate": 0,
        "detect_car": 0,
        "extract_features": 0,
        "predict_scene": 0,
        "simple_photo_describe": 0,
        "photo_describe": 0,
    }

    class ImageLoader:
        def __init__(self, path, source):
            counters["ImageLoader"] += 1
            self.path = path
            self.source = source

        def load_image(self):
            return loaded_image(self.path) if callable(loaded_image) else loaded_image

    class ImageRotator:
        def __init__(self):
            counters["ImageRotator"] += 1

        def rotate_image(self, image):
            counters["rotate"] += 1
            return image

    class CarDetector:
        def __init__(self):
            counters["CarDetector"] += 1

        def detect_car(self, image):
            counters["detect_car"] += 1
            return has_car

    class ObjectExtractor:
        def __init__(self):
            counters["ObjectExtractor"] += 1

        def extract_features(self, image):
            counters["extract_features"] += 1
            return {"Tree": 2, "Building": 1}

    class SceneExtractor:
        def __init__(self):
            counters["SceneExtractor"] += 1

        def predict_scene(self, image):
            counters["predict_scene"] += 1
            # main.py removes the last item (comment says it is "tunnel").
            return {"city street": 0.81, "tunnel": 1.0}

    class PhotoDescriber:
        def __init__(self):
            counters["PhotoDescriber"] += 1

        def simple_photo_describe(self, image):
            counters["simple_photo_describe"] += 1
            return {
                "base": "base",
                "detailed": "detailed",
                "alternative": "alternative",
            }

    class PhotoDescriberWithQuestion:
        def __init__(self):
            counters["PhotoDescriberWithQuestion"] += 1

        def photo_describe(self, image):
            counters["photo_describe"] += 1
            return "vqa"

    fake_autocaption = types.ModuleType("autocaption")
    fake_autocaption.ImageLoader = ImageLoader
    fake_autocaption.ImageRotator = ImageRotator
    fake_autocaption.CarDetector = CarDetector
    fake_autocaption.ObjectExtractor = ObjectExtractor
    fake_autocaption.SceneExtractor = SceneExtractor
    fake_autocaption.PhotoDescriber = PhotoDescriber
    fake_autocaption.PhotoDescriberWithQuestion = PhotoDescriberWithQuestion

    monkeypatch.setitem(sys.modules, "autocaption", fake_autocaption)

    main = load_source_module("_main_under_test", "main.py")
    return main, counters


def test_pipeline_success(monkeypatch):
    main, counters = load_main_with_fakes(
        monkeypatch,
        loaded_image=object(),
        has_car=True,
    )

    result = main.run_pipeline(["car.jpg"], source=True)

    assert result == [[
        {"Tree": 2, "Building": 1},
        {"city street": 0.81},
        "Базовое описание: base",
        "Подробное описание: detailed",
        "Альтернативное описание: alternative",
        "Описание по вопросу (VQA): vqa",
    ]]

    assert counters["ImageLoader"] == 1
    assert counters["rotate"] == 1
    assert counters["detect_car"] == 1
    assert counters["extract_features"] == 1
    assert counters["predict_scene"] == 1
    assert counters["simple_photo_describe"] == 1
    assert counters["photo_describe"] == 1


def test_pipeline_skips_remaining_steps_when_image_cannot_be_loaded(monkeypatch):
    main, counters = load_main_with_fakes(
        monkeypatch,
        loaded_image=None,
        has_car=True,
    )

    result = main.run_pipeline(["missing.jpg"], source=True)

    assert result == [[
        "Не удалось загрузить изображение: missing.jpg."
    ]]
    assert counters["rotate"] == 0
    assert counters["detect_car"] == 0
    assert counters["extract_features"] == 0
    assert counters["predict_scene"] == 0
    assert counters["simple_photo_describe"] == 0
    assert counters["photo_describe"] == 0


def test_pipeline_stops_when_car_is_not_detected(monkeypatch):
    main, counters = load_main_with_fakes(
        monkeypatch,
        loaded_image=object(),
        has_car=False,
    )

    result = main.run_pipeline(["no-car.jpg"], source=True)

    assert result == [[
        "Не удалось определить наличие автомобиля на изображении: no-car.jpg."
    ]]
    assert counters["rotate"] == 1
    assert counters["detect_car"] == 1
    assert counters["extract_features"] == 0
    assert counters["predict_scene"] == 0
    assert counters["simple_photo_describe"] == 0
    assert counters["photo_describe"] == 0


def test_heavy_components_are_initialized_once_for_multiple_images(monkeypatch):
    main, counters = load_main_with_fakes(
        monkeypatch,
        loaded_image=object(),
        has_car=True,
    )

    main.run_pipeline(["one.jpg", "two.jpg"], source=True)

    assert counters["ImageRotator"] == 1
    assert counters["CarDetector"] == 1
    assert counters["ObjectExtractor"] == 1
    assert counters["SceneExtractor"] == 1
    assert counters["PhotoDescriber"] == 1
    assert counters["PhotoDescriberWithQuestion"] == 1

    assert counters["ImageLoader"] == 2
    assert counters["rotate"] == 2
    assert counters["detect_car"] == 2
