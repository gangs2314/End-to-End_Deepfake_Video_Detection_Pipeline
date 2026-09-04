import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import os

# Labels
LABELS = ["REAL", "AI_GENERATED"]

# Load model
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "resnet18.pth")

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return LABELS[prediction]

# TEST IMAGE (CHANGE THIS PATH)
test_image = os.path.join(
    PROJECT_ROOT,
    "data",
    "frames",
    "ai_generated",
    os.listdir(os.path.join(PROJECT_ROOT, "data", "frames", "ai_generated"))[0]
)

result = predict(test_image)
print("Prediction:", result)
