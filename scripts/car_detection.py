from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import io
import collections
import concurrent.futures
import pandas as pd
import os
from tqdm import tqdm

def load_custom_yolo_model():
    model_path = 'runs/detect/train3/weights/best.pt'
    model = YOLO(model_path)
    return model

def load_orientation_model():
    response = requests.get('https://github.com/StalSkyle/car-scene-captioning/raw/main/models/resnet50_rotation_car_99.76.pth')
    response.raise_for_status()
    model_data = torch.load(io.BytesIO(response.content), map_location='cpu', weights_only=False)
    if isinstance(model_data, collections.OrderedDict):
        import torchvision.models as models
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(model_data)
        model.eval()
        return model
    else:
        model_data.eval()
        return model_data


model = load_custom_yolo_model()
orientation_model = load_orientation_model()

angle_values = {0: 0, 1: 90, 2: 180, 3: 270}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception:
        return None

def detect_car_simple(url):
    image = load_image(url)
    if image is None:
        return False
    
    # Определяем ориентацию изображения
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = orientation_model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    rotation_angle = angle_values[predicted_class]
    rotated_image = image.rotate(-rotation_angle, expand=True)
    
    result = model(rotated_image, verbose=False)  
    
    if result[0].boxes is None:
        return False
        
    for box in result[0].boxes:
        class_id = int(box.cls[0])
        confidence = box.conf[0].item()
        
        class_name = model.names[class_id]
        
        if class_name in ['car', 'truck'] and confidence > 0.25:  
            return True
    
    return False

def create_stratified_sample(df, sample_size=50000, random_state=42):
    try:
        sample = df.groupby('city', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // df['city'].nunique()), 
                                random_state=random_state)
        )
        return sample
    except Exception as e:
        sample = df.sample(min(sample_size, len(df)), random_state=random_state)
        print(f"Случайная выборка (fallback): {len(sample)} изображений")
        return sample

def process_batch(urls_batch):
    results = []
    for url in urls_batch:
        try:
            results.append(detect_car_simple(url))
        except Exception as e:
            results.append(False)
    return results

def smart_processing_pipeline(df, target_sample_size=50000, max_workers=8, batch_size=50):    
    if len(df) > target_sample_size:
        df_work = create_stratified_sample(df, target_sample_size)
    else:
        df_work = df.copy()
        print(f"Работаем со всем датасетом: {len(df_work)} изображений")
    
    print(f"Будет обработано: {len(df_work)} изображений")
    print(f"Параллельных workers: {max_workers}")
    print(f"Размер батча: {batch_size}")

    urls = df_work['url'].tolist()
    
    batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
    print(f"Создано батчей: {len(batches)}")
    
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                          total=len(batches), 
                          desc="Обработка батчей"):
            batch_results = future.result()
            all_results.extend(batch_results)
    
    df_work = df_work.copy()
    df_work['car_detected'] = all_results
    
    car_count = df_work['car_detected'].sum()
    car_ratio = car_count / len(df_work)
    
    print(f"\nРЕЗУЛЬТАТЫ ОБРАБОТКИ:")
    print(f"   Обработано: {len(df_work)} изображений")
    print(f"   С машинами: {car_count} ({car_ratio*100:.1f}%)")
    print(f"   Без машин: {len(df_work) - car_count} ({(1-car_ratio)*100:.1f}%)")
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output_file = f'car_detection_results_{timestamp}.csv'
    df_work.to_csv(output_file, index=False)
    print(f"Результаты сохранены в: {output_file}")
    
    return df_work

# Запускаем обработку
print("Запускаем обработку с ДООБУЧЕННОЙ моделью...")
df = pd.read_csv('output.csv')
df_result = smart_processing_pipeline(
    df=df,
    target_sample_size=50000,
    max_workers=8,
    batch_size=50
)

OUTPUT_DIR = "dataset_car_detected"
os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(OUTPUT_DIR, "dataset_car_detected.csv")
df_result.to_csv(save_path, index=False)
print(f"Финальные результаты сохранены в: {save_path}")
