import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Подготовка модели поворота
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../MODELS/resnet50_rotation_car_99.76.pth"
angle_values = {0: 0, 1: 90, 2: 180, 3: 270}

rotation_model = models.resnet50(weights=None)
num_features = rotation_model.fc.in_features
rotation_model.fc = nn.Linear(num_features, 4)
rotation_model.load_state_dict(torch.load(model_path, map_location=device))
rotation_model = rotation_model.to(device)
rotation_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def rotate_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = rotation_model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    rotation_angle = angle_values[predicted_class]
    rotated_image = image.rotate(-rotation_angle, expand=True)
    return rotated_image

# Загрузка YOLO модели
try:
    yolo_model = YOLO('MODELS/yolov8x-oiv7.pt')
except Exception as e:
    raise RuntimeError("Скачайте модельку с Jupyter Lab, в GitHub не помещается") from e

# Поиск первой подходящей картинки
image_dir = Path('../../good_images')
image_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
image_path = None

for img_path in image_dir.iterdir():
    if img_path.suffix.lower()[1:] in image_extensions:
        image_path = img_path
        break

if image_path is None:
    raise FileNotFoundError("Не найдено изображений с расширениями: jpg, jpeg, png, bmp в папке ../DATASETS/4K")

print(f"Обрабатывается изображение: {image_path}")

# Поворот изображения
rotated_img = rotate_image(str(image_path))

# Применение YOLO
results = yolo_model(rotated_img, verbose=False)
result = results[0]

# Визуализация результата
annotated_img = result.plot()  # возвращает numpy array (BGR -> RGB для matplotlib)
annotated_img_rgb = annotated_img[..., ::-1]  # конвертация BGR в RGB

plt.figure(figsize=(12, 8))
plt.imshow(annotated_img_rgb)
plt.axis('off')
plt.title(f"Результат YOLO на изображении: {image_path.name}")
plt.show()

# Вывод детектированных классов
detected_classes = set()
for box in result.boxes:
    cls_id = int(box.cls)
    cls_name = result.names[cls_id]
    detected_classes.add(cls_name)

print("\nДетектированные классы на изображении:")
for cls in sorted(detected_classes):
    print(f"- {cls}")