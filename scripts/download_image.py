import os
import requests
import pandas as pd
from tqdm import tqdm

CSV_PATH = "after_ride_photos_fin.csv"
OUTPUT_DIR = "dataset_images" # куда сохранять изображения

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, sep='\t')
df = df[0:200] # можете менять
urls = df["url"].dropna().unique()

print(f"Найдено {len(urls)} URL. Начинаем загрузку...")

for i, url in enumerate(tqdm(urls, desc="Downloading images")):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image/" in content_type:
            ext = content_type.split("/")[-1].split(";")[0]
        else:
            ext = "jpg"

        save_path = os.path.join(OUTPUT_DIR, f"img_{i:05d}.{ext}")
        with open(save_path, "wb") as f:
            f.write(response.content)

    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")

print("Картинки сохранены в:", OUTPUT_DIR)

