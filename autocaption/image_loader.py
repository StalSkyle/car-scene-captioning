from PIL import Image
from io import BytesIO
import requests

class ImageLoader:
    def __init__(self, path: str, source: bool = True):
        """
        Args:
            path: путь до изображения (URL или локальный путь)
            source: 'local' (True) или 'url' (False)
        """
        self.path = path
        self.source = source
        self.image = None

    def load_image(self):
        """Загружает изображение из локального пути или URL."""
        try:
            if self.source:
                self.image = Image.open(self.path).convert("RGB")
            else:
                response = requests.get(self.path, timeout=10)
                response.raise_for_status()
                self.image = Image.open(BytesIO(response.content)).convert("RGB")
            return self.image
        except Exception as e:
            print(f"Ошибка при загрузке изображения {self.path}: {e}")
            return None
