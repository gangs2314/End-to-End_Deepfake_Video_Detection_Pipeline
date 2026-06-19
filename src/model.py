import torch
import torch.nn as nn
import torchvision.models as models


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load("model/resnet18.pth", map_location="cpu")
    )

    model.eval()
    return model
