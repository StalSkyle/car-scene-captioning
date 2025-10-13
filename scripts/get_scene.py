#TODO: пожалуйста, не ругайте меня за этот говнокод, я всё красиво перепишу и отрефакторю , честно

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# TODO: something something don't repeat yourself. оформить в виде класса, сделать вращение методом класса
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

# TODO: ну что это такое...
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

# получаем инференс
def predict_scene(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze()  # multi-label

    labels = [
        "other",
        "highway",
        "residential",
        "city street",
        "parking lot",
        "gas stations",
        "tunnel"
    ]
    return {lbl: float(p) for lbl, p in zip(labels, probs)}

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 7)

checkpoint = torch.load('../MODELS/resnet18_4x_scene_tag_bdd100k.pth', map_location='cpu', weights_only=True)
state_dict = checkpoint['state_dict']

# убираем префиксы 'backbone.' и заменяем 'head.fc' на 'fc'
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('backbone.'):
        new_key = k[len('backbone.'):]
        new_state_dict[new_key] = v
    elif k == 'head.fc.weight':
        new_state_dict['fc.weight'] = v
    elif k == 'head.fc.bias':
        new_state_dict['fc.bias'] = v

# загружаем
model.load_state_dict(new_state_dict, strict=True)
model.eval()

# предиктим
for i in range(1000):
    try:
        # это я тестил
        img_path = f"../../../dataset_images/img_{(5 - len(str(i))) * '0' + str(i)}.jpeg"
        rotated_img = rotate_image(img_path)
        result = predict_scene(rotated_img, model)

        print(f"good_images/img_{(5 - len(str(i))) * '0' + str(i)}.jpeg")
        print(result)
    except:
        pass
