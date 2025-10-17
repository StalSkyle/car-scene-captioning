# TODO: мы не успели внедрить LLaVa в проект, но она работает отдельным скриптом
# TODO: простите за захардкоженность картинки и промта, скрипт доделывался за три часа до защиты
# TODO: данный скрипт работает на torch=2.9.0, transformers=4.57.1. мы не тестили, работает ли он с прописанными requirements
# картинка: https://drive.google.com/file/d/1_6lCXKqckk5HSVDuVDbw_ntxzPJje3x-/view?usp=sharing

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Путь к изображению и промпт
IMAGE_PATH = "img_00001.jpeg"
PROMPT = """
USER: <image>\n
Describe what you see in this image. Focus on the surroundings of the car.
You also have outputs of models that tried to capture some information. Use it if you want; they might be wrong.

Object detection model {'Car': 0.34714019298553467, 'Wheel': 0.3243175148963928}
Scene classification model {'other': 0.08122928440570831, 'highway': 0.12126690149307251, 'residential': 0.8982969522476196, 'city street': 0.06356442719697952, 'parking lot': 0.7530761361122131, 'gas stations': 0.11552013456821442}
Basic description: a car parked in a parking lot
Longer description: a car parked in a busy parking lot
Alternative description: a parked vehicle in a parking lot
Other description (VQA): parking lot
\n
ASSISTANT:"""

# Загрузка модели и процессора
print("Loading LLaVA model...")
model_id = "llava-hf/llava-1.5-7b-hf"  # или "llava-hf/llava-1.5-13b-hf"


model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(model_id)

image = Image.open(IMAGE_PATH).convert("RGB")

inputs = processor(text=PROMPT, images=image, return_tensors="pt").to(
    model.device)

# Генерация ответа
print("Generating description...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
    )

# Декодирование результата
generated_text = processor.decode(output[0], skip_special_tokens=True)
print("\n" + "=" * 50)
print("LLaVA Response:")
print(generated_text)
print("=" * 50)

"""
Вывод модели

The image features a silver car parked in a parking lot. 
The car is positioned in the middle of the parking lot, and there are several other cars parked around it.
In the background, there are a few trucks and a bus, indicating that the parking lot is located near a busy area. 
The scene also includes a traffic light, suggesting that the parking lot is situated near an intersection or a busy street
"""