import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from model import load_model

def classify_video(frames_dir):
    model = load_model()
    model.eval()

    # Must match dataset.py transform exactly
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    real_scores = []

    for img_name in os.listdir(frames_dir):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(frames_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
            
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            prob = torch.softmax(output, dim=1)

            # DATASET MAPPING: 0 = Real, 1 = Fake
            # So index 0 is the probability of being REAL
            p_real = prob[0][0].item()
            real_scores.append(p_real)

    if len(real_scores) == 0:
        return "ERROR: NO FACE DETECTED", 0.0

    avg_real_score = float(np.mean(real_scores))
    
    # LOGIC:
    # If the average probability of being "Real" is high (> 50%), it's Real.
    if avg_real_score > 0.50:
        result = "REAL"
    else:
        result = "AI_GENERATED"

    return result, avg_real_score