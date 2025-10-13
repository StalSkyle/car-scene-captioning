import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import io
from ultralytics import YOLO
import requests
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms
import collections
import concurrent.futures
from tqdm.auto import tqdm
import os
from pathlib import Path
import random

os.makedirs('images/')
os.makedirs('labels/')

# Создаем директорию для моделей
model_dir = Path('/tmp/yolo_models')
model_dir.mkdir(exist_ok=True)

model_path = model_dir / 'yolov8n.pt'

# Скачиваем модель вручную если нет
if not model_path.exists():
    print("Скачиваем YOLO модель...")
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Модель успешно скачана!")
    except Exception as e:
        print(f"Ошибка скачивания: {e}")

# Теперь загружаем модель
from ultralytics import YOLO
model = YOLO(str(model_path))
print("Модель загружена!")

DATASET_URL = "https://datasets-server.huggingface.co/rows?dataset=yandex%2Fmad-cars&config=default&split=train"
# file_path = 'project/Artem/'
NUM_SAMPLES = 500 
VAL_RATIO = 0.2 

def get_samples():
    total_size = 5800000
    random_indices = random.sample(range(total_size), min(NUM_SAMPLES, total_size))
    random_indices.sort()

    success_count = 0
    batch_size = 100
    with tqdm (total=len(random_indices), desc="Скачивание данных") as pbar:
        num = 0
        for i in range(0, len(random_indices), batch_size):
            batch_indices = random_indices[i:i + batch_size]
            
            try:
                url = f"{DATASET_URL}&length={len(batch_indices)}"
                for j, idx in enumerate(batch_indices):
                    if j == 0:
                        url += f"&offset={idx}"
                response = requests.get(url, timeout=60)
                data = response.json()
                rows = data.get('rows', [])
                
                # Обрабатываем каждый элемент в батче
                for row_idx, row in enumerate(rows):
                    try:
                        row_data = row.get('row', {})
                        
                        # Извлекаем метаданные
                        image_url = row_data.get('url', '')  # URL изображения
                        img_response = requests.get(image_url, timeout=30)
                        image = Image.open(io.BytesIO(img_response.content))
                        filename = f'img{num}'
                        num += 1
                        img_path = f'images/{filename}.jpg'
                        image.save(img_path, 'JPEG')
                        
                        img_width, img_height = image.size
                        label_path = f'labels/{filename}.txt'
                        
                        with open(label_path, 'w') as f:
                            f.write("0 0.5 0.5 1.0 1.0\n")
                    except Exception as e:
                        print(f"Error {e}")
                        continue
                    
                    pbar.update(1)
                        
            except Exception as e:
                print(f"Ошибка: {e}")
                continue
                        
def create_dataset_yaml():
    """Создаем конфиг файл"""
    yaml_content = """path: 
train: images
val: images

names:
  0: car
"""
    with open(f'cars_dataset.yaml', 'w') as f:
        f.write(yaml_content)  

get_samples()
create_dataset_yaml()
def train_simple():
    # Обучаем
    results = model.train(
        data='cars_dataset.yaml',
        epochs=30,          
        imgsz=640,
        batch=16,
        lr0=0.001,
        patience=10,         
        save=True,
        verbose=True
    )
    
    return results

train_results = train_simple()
