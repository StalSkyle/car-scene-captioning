class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    def __init__(self, class_id):
        self.cls = FakeScalar(class_id)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def __call__(self, image, verbose=False):
        self.calls.append((image, verbose))
        return self.results


def make_detector(module, results):
    detector = module.CarDetector.__new__(module.CarDetector)
    detector.car_detection_model = FakeModel(results)
    return detector


def test_detect_car_returns_true_for_car(image_processor_module):
    result = FakeResult(
        boxes=[FakeBox(0)],
        names={0: "Car"},
    )
    detector = make_detector(image_processor_module, [result])
    image = object()

    assert detector.detect_car(image) is True
    assert detector.car_detection_model.calls == [(image, False)]


def test_detect_car_returns_true_for_truck(image_processor_module):
    result = FakeResult(
        boxes=[FakeBox(3)],
        names={3: "TRUCK"},
    )
    detector = make_detector(image_processor_module, [result])

    assert detector.detect_car(object()) is True


def test_detect_car_returns_false_when_no_boxes(image_processor_module):
    result = FakeResult(boxes=[], names={})
    detector = make_detector(image_processor_module, [result])

    assert detector.detect_car(object()) is False


def test_detect_car_returns_false_for_other_object(image_processor_module):
    result = FakeResult(
        boxes=[FakeBox(1), FakeBox(2)],
        names={1: "person", 2: "bicycle"},
    )
    detector = make_detector(image_processor_module, [result])

    assert detector.detect_car(object()) is False
