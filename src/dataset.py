import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []

        # 0 = Real, 1 = Fake (includes ai_generated and ai_edited)
        self.classes = {
            "real": 0,
            "ai_generated": 1,
            "ai_edited": 1  # Add this to train on edited videos too
        }

        print(f"Scanning dataset in {root_dir}...")
        for cls_name, label in self.classes.items():
            folder = os.path.join(root_dir, cls_name)
            if os.path.exists(folder):
                count = 0
                for img in os.listdir(folder):
                    if img.endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(folder, img))
                        self.labels.append(label)
                        count += 1
                print(f"  Loaded {count} images from {cls_name}")
            else:
                print(f"  Warning: Folder {cls_name} not found.")

        # ADD NORMALIZATION to match inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = self.transform(image)
        label = self.labels[index]
        return image, label