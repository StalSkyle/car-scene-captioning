import os
import sys
import json
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.transforms.functional import InterpolationMode

# === Настройки ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROTATION_MODEL_PATH = ".../MODELS/resnet50_rotation_car_99.76.pth"
BLIP_DIR = "/home/jupyter/project/Grisha/BLIP"
IMAGE_FOLDER = "/home/jupyter/project/Grisha/dataset_images"
OUTPUT_JSON = "/home/jupyter/project/Grisha/blip_vqa_environment.json"

# Углы поворота
angle_values = {0: 0, 1: 90, 2: 180, 3: 270}

# === Клонирование BLIP (если нужно) ===
if not os.path.exists(BLIP_DIR):
    print("Клонируем BLIP...")
    os.system(f"git clone https://github.com/salesforce/BLIP.git {BLIP_DIR}")

if BLIP_DIR not in sys.path:
    sys.path.insert(0, BLIP_DIR)

from models.blip_vqa import blip_vqa


# === Загрузка модели поворота ===
def load_rotation_model():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)
    model.load_state_dict(torch.load(ROTATION_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


rotation_model = load_rotation_model()


# === Поворот изображения ===
def correct_rotation(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = rotation_model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    rotation_angle = angle_values[predicted_class]
    rotated_image = image.rotate(-rotation_angle, expand=True)
    return rotated_image


# === Загрузка BLIP VQA ===
med_config = os.path.join(BLIP_DIR, "configs", "med_config.json")
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'

print("Загружаем BLIP VQA...")
model = blip_vqa(pretrained=model_url, image_size=480, vit='base', med_config=med_config)
model.eval()
model = model.to(device)
print("✅ BLIP VQA загружена")


# === Загрузка изображения для BLIP ===
def load_blip_image(pil_image, image_size=480):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
    ])
    return transform(pil_image).unsqueeze(0).to(device)


# === Вопрос ===
QUESTION = "Describe only the surroundings near the car, such as buildings, trees, street lights, signs, or pavement. Do not mention the car, vehicle, sky, or clouds."


# === Обработка одного изображения ===
def process_image(image_path):
    try:
        # 1. Поворот
        rotated = correct_rotation(image_path)
        # 2. Подготовка для BLIP
        image_tensor = load_blip_image(rotated)
        # 3. VQA
        with torch.no_grad():
            answer = model(image_tensor, [QUESTION], train=False, inference='generate')
        return answer[0]
    except Exception as e:
        return f"Ошибка: {str(e)}"


# === Обработка папки ===
results = {}
files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Найдено изображений: {len(files)}")

for i, fname in enumerate(files):
    path = os.path.join(IMAGE_FOLDER, fname)
    print(f"[{i + 1}/{len(files)}] {fname}")
    desc = process_image(path)
    results[fname] = desc
    print(f"  → {desc}")

# === Сохранение ===
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\n✅ Готово! Результаты в: {OUTPUT_JSON}")