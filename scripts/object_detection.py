import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# подготовка модели, которая вращает картинку
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


# функция, которая вращает картинку
def rotate_image(image_path):
    image = Image.open(image_path).convert('RGB')

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = rotation_model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    rotation_angle = angle_values[predicted_class]
    rotated_image = image.rotate(-rotation_angle, expand=True)

    return rotated_image


# размечаем
try:
    yolo_model = YOLO('../MODELS/yolov8x-oiv7.pt')
except:
    raise 'скачайте модельку с jupyter lab, в гитхаб не помещается'

image_dir = Path('../DATASETS/4K')
class_image_count = {}
image_extensions = {'jpg', 'jpeg', 'png', 'bmp'}

for img_path in tqdm(image_dir.iterdir()):

    if str(img_path).split('.')[-1] not in image_extensions:
        continue
    # получаем повёрнутое изображение
    rotated_img = rotate_image(str(img_path))

    # прогоняем через YOLO
    results = yolo_model(rotated_img,
                         verbose=False)
    result = results[0]

    # собираем уникальные классы на изображении
    detected_classes = set()
    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = result.names[cls_id]
        detected_classes.add(cls_name)
        conf = float(box.conf)

    # обновляем статистику
    for cls in detected_classes:
        class_image_count[cls] = class_image_count.get(cls, 0) + 1

# вывод
print("=" * 50)
print("Статистика по классам (сколько изображений содержат класс):")
for cls, count in sorted(class_image_count.items(), key=lambda item: item[1])[
    ::-1]:
    print(f"{cls:<25}: {count}")

'''
==================================================
Статистика по классам (сколько изображений содержат класс):
Car                      : 3879
Wheel                    : 3074
Tire                     : 1625
Vehicle registration plate: 1597
Van                      : 183
Tree                     : 126
Window                   : 80
Street light             : 52
Footwear                 : 41
Person                   : 39
Land vehicle             : 25
Man                      : 23
Plant                    : 22
Truck                    : 20
Building                 : 17
Clothing                 : 16
Woman                    : 16
Jeans                    : 14
Taxi                     : 10
Bus                      : 6
Traffic sign             : 5
Bicycle                  : 4
Bench                    : 4
Billboard                : 3
Girl                     : 2
Flag                     : 2
Waste container          : 2
House                    : 1
Airplane                 : 1
Watch                    : 1
Bicycle wheel            : 1
Skyscraper               : 1
Computer mouse           : 1
Segway                   : 1
Vehicle                  : 1
Mirror                   : 1
Shorts                   : 1
Suitcase                 : 1
Motorcycle               : 1
Jacket                   : 1
Human face               : 1
Luggage and bags         : 1
'''
