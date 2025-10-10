import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 1. Настройки
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../models/resnet50_rotation_car_99.76.pth"
image_path = "car.jpeg"  # ПУТЬ К ИЗОБРАЖЕНИЮ
output_image_path = "rotated_car.png"  # ПУТЬ, КУДА СОХРАНИТЬ ГОТОВОЕ ИЗОБРАЖЕНИЕ

# angles = {0: "0", 1: "90", 2: "180", 3: "270"}
angle_values = {0: 0, 1: 90, 2: 180, 3: 270}

# 2. Загрузка модели
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 3. Подготовка изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# 4. Предсказание
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

# Вывод результата (для дебага)
# print(f"Предсказанный угол: {angles[predicted_class]}")
# print(f"Уверенность: {confidence:.4f} ({confidence * 100:.2f}%)")


# 5. Поворот изображения и сохранение
rotation_angle = angle_values[predicted_class]
rotated_image = image.rotate(-rotation_angle, expand=True)
rotated_image.save(output_image_path)

# print(f"Повернутое изображение сохранено как: {output_image_path}")
