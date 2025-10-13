# TODO: добавить поворот картинки

from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch

# получаем инференс
def predict_scene(image_path, model):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(img).unsqueeze(0)

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
model.fc = torch.nn.Linear(model.fc.in_features, 7)  # ← БЫЛО 6, СТАЛО 7

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
        result = predict_scene(f"../../../good_images/img_{(5 - len(str(i))) * '0' + str(i)}.jpeg", model)

        print(f"good_images/img_{(5 - len(str(i))) * '0' + str(i)}.jpeg")
        print(result)
    except:
        pass
