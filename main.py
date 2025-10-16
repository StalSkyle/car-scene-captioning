from autocaption import ImageLoader, ImageRotator, CarDetector, \
    ObjectExtractor, SceneExtractor, PhotoDescriber, PhotoDescriberWithQuestion


def run_pipeline(image_path: list[str], source: bool) -> list:
    """
    Прогоняет картинку через модели с помощью написанной библиотеки autocaption
    :param image_path: путь к изображению
    :param source: источник картинки (True = локальный путь, False = URL)
    :return: текстовое описание окружения картинки
    """
    res = [[] * len(image_path)]

    # инициализация классов
    print('Инициализация моделей, пожалуйста, подождите...')
    print('Инициализация модели поворота')
    rotator = ImageRotator()
    print('Инициализация модели детектора автомобилей')
    car_detector = CarDetector()
    print('Инициализация модели нахождения объектов')
    object_extractor = ObjectExtractor()
    print('Инициализация модели классификации сцены')
    scene_classificator = SceneExtractor()
    print('Инициализация модели BLIP')
    describer = PhotoDescriber()
    print('Инициализация модели BLIP с вопросом в промте')
    q_describer = PhotoDescriberWithQuestion()

    # поехали
    print('Начинается обработка изображений')
    for i, path in enumerate(image_path):
        image = ImageLoader(path=path, source=source).load_image()

        if image is None:
            res[i].append(f"Не удалось загрузить изображение: {path}.")
            continue

        # 1. Обработка изображения

        image = rotator.rotate_image(image)
        if not car_detector.detect_car(image):
            res[i].append(
                f"Не удалось определить наличие автомобиля на изображении: {path}.")
            continue

        # 2. Нахождение признаков

        # нахождение предметов на фотографии
        res[i].append(object_extractor.extract_features(image))

        # определение сцены

        # убрал класс tunnel, так как там модель почти всегда выдает 1
        temp_dict = scene_classificator.predict_scene(image)
        temp_dict.popitem()

        res[i].append(temp_dict)

        # генерация краткого описания
        result = describer.simple_photo_describe(image)
        res[i].append(f"Базовое описание: {result['base']}")
        res[i].append(f"Подробное описание: {result['detailed']}")
        res[i].append(f"Альтернативное описание: {result['alternative']}")

        # Получаем дополнительно описание через BLIP VQA с конкретным вопросом

        vqa_desc = q_describer.photo_describe(image)
        res[i].append(f"Описание по вопросу (VQA): {vqa_desc}")

    return res


if __name__ == "__main__":
    # тесты
    res = run_pipeline(
        ["/home/jupyter/project/Grisha/dataset_images/img_00000.jpeg"],
        source=True)
    # res = run_pipeline(
    #     ['https://carsharing-acceptances.s3.yandex.net/000080b5-5f40-4f6b-56bd-68e42cfe0f1d/car_location_22075860-8af6-11f0-b981-052fecb11955.CAP84636423206376342.jpg/46f3928-fad162a0-af3ae0d5-516d116e'],
    #     source=False)

    for i in range(len(res[0])):
        print(res[0][i])
