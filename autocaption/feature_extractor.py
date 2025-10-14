class FeatureExtractor:
    def __init__(self, model_name="resnet50", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def init_model(self):
        """Загружает предобученную модель для извлечения признаков."""
        pass

    def extract_features(self, image):
        """Извлекает признаки из изображения."""
        pass