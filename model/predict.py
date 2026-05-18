import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from io import BytesIO

MODEL_PATH = "model/plant_model.pth"
CLASSES_PATH = "model/classes.txt"

with open(CLASSES_PATH) as f:
    class_names = [l.strip() for l in f.readlines()]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

def predict(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    top3_probs, top3_idx = torch.topk(probs, 3)

    return {
        "predicted_class": class_names[top3_idx[0].item()],
        "confidence": round(top3_probs[0].item() * 100, 2),
        "top3": [
            {"class": class_names[i.item()], "confidence": round(p.item() * 100, 2)}
            for p, i in zip(top3_probs, top3_idx)
        ]
    }