import torch
from torch import nn, optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
import os

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_PATH = os.path.join(PROJECT_ROOT, "data", "frames")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model")
os.makedirs(MODEL_PATH, exist_ok=True)

# Dataset
dataset = DeepfakeDataset(FRAME_PATH)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f}")

# Save model
torch.save(model.state_dict(), os.path.join(MODEL_PATH, "resnet18.pth"))
print("✅ Model training complete. Saved as model/resnet18.pth")
