from autocaption import ImageLoader, ImageProcessor, ObjectExtractor, SceneExtractor


def run_pipeline(image_path: list[str], source: bool) -> list:
    """
    Прогоняет картинку через модели с помощью написанной библиотеки autocaption
    :param image_path: путь к изображению
    :param source: источник картинки (True = локальный путь, False = URL)
    :return: текстовое описание окружения картинки
    """
    res = [[] * len(image_path)]

    # инициализация классов

    processor = ImageProcessor()
    object_extractor = ObjectExtractor()
    scene_classificator = SceneExtractor()

    # поехали
    for i, path in enumerate(image_path):
        image = ImageLoader(path=path, source=source).load_image()

        if image is None:
            res[i] = f"Не удалось загрузить изображение: {path}."
            continue

        res[i] = dict()

        # 1. Обработка изображения

        image = processor.rotate_image(image)

        # 2. Нахождение признаков

        # нахождение предметов на фотографии
        res[i]['детекция объектов'] = object_extractor.extract_features(image)

        # определение сцены
        res[i]['сцена'] = scene_classificator.predict_scene(image)

    return res


if __name__ == "__main__":
    # тесты
    # print(run_pipeline(['../../parking.png'], source=True))
    print(run_pipeline(['https://carsharing-acceptances.s3.yandex.net/000080b5-5f40-4f6b-56bd-68e42cfe0f1d/car_location_22075860-8af6-11f0-b981-052fecb11955.CAP84636423206376342.jpg/46f3928-fad162a0-af3ae0d5-516d116e'], source=False))
