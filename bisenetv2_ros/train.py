import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----------------------------
# 0. Config logging
# ----------------------------
#### On enregistre les log dans un fichier 
logging.basicConfig(filename="train.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# ----------------------------
# 1. Utilitaires
# ----------------------------

##### Charge le fichier de couleurs d'annotations 
def load_colormap(path):
    colormap = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:  # Doit avoir [index, nom, R, G, B]
                continue
            try:
                index = int(parts[0])
                r, g, b = map(int, parts[2:5])  # R, G, B aux indices 2, 3, 4
                colormap[(r, g, b)] = index
            except ValueError:
                continue  # Ignore les lignes mal formatées
    return colormap
    
### Transforme une carte de labels en une image RGB 
def decode_segmap(label, colormap):
    h, w = label.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in enumerate(colormap):
        color_mask[label == cls] = color
    return Image.fromarray(color_mask)
    
### Convertit l'image masque en masques d'indices de classes utilisables    
def rgb_to_mask(mask, colormap): 
    mask = np.asarray(mask)
    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, idx in colormap.items():
        matches = np.all(mask == rgb, axis=-1)
        label_mask[matches] = idx
    return label_mask

# ----------------------------
# 2. Dataset RUGD
# ----------------------------
## On custom le dataset 

class RUGDDataset(Dataset):
    def __init__(self, txt_file, transform=None, colormap=None):
        self.pairs = [line.strip().split() for line in open(txt_file)]
        self.transform = transform
        self.colormap = colormap

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = np.asarray(Image.open(img_path).convert('RGB'))
        mask = np.asarray(Image.open(mask_path).convert('RGB'))
        

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        mask = rgb_to_mask(mask, self.colormap)
        mask = torch.from_numpy(mask).long()

        return img, mask


# ----------------------------
# 3. mIoU
# ----------------------------

### Retourne la liste des IoU
def compute_iou(pred, target, num_classes):
    ious = []
    pred = pred.flatten()
    target = target.flatten()
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

# ----------------------------
# 4. Validation
# ----------------------------
def validate(model, val_loader, num_classes, device):
    model.eval()
    iou_list = []
    correct = 0
    total = 0

    with torch.no_grad():
    ## Modèle en mode validation
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)[0]
            preds = torch.argmax(outputs, dim=1)

            for pred, label in zip(preds, labels):
                ious = compute_iou(pred.cpu().numpy(), label.cpu().numpy(), num_classes)
                iou_list.append(ious)
                correct += (pred == label).sum().item()
                total += label.numel()
## Calcule mIoU et pixel accuracy
    iou_arr = np.array(iou_list)
    mean_iou = np.nanmean(iou_arr, axis=0)
    miou = np.nanmean(mean_iou)
    acc = correct / total

    print(f"[Validation] mIoU: {miou:.4f} | Pixel Accuracy: {acc:.4f}")
    logging.info(f"[Validation] mIoU: {miou:.4f} | Pixel Accuracy: {acc:.4f}")
    return miou, acc

# ----------------------------
# 5. Entraînement principal ## Modification taille de batch, etc. ici
# ----------------------------
def train():
    from lib.models.bisenetv2 import BiSeNetV2  #télécharger le modèle au préalable

    # Caractéristiques du modèle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    num_classes = 25
    batch_size=8         
    shuffle=True
    num_workers=12
    pin_memory=True
    num_epochs = 50
    lr = 1e-2

    # Chemins
    base_path = 'RUGD'
    colormap_path = os.path.join(base_path, 'RUGD_annotation-colormap.txt')
    train_list = os.path.join(base_path, 'train.txt')
    val_list = os.path.join(base_path, 'val.txt')
    
    # Transformations avec Albumentations
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop((512, 512), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    colormap = load_colormap(colormap_path)

    train_set = RUGDDataset(train_list, transform=train_transform, colormap=colormap)
    val_set = RUGDDataset(val_list, transform=val_transform, colormap=colormap)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = BiSeNetV2(n_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("res", exist_ok=True)
    
    # Entrainement par epoch 
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), f"res/model_epoch{epoch+1}.pth")
        epoch_loss = running_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1} finished. Loss: {epoch_loss:.4f}")
        logging.info(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # Validation
        validate(model, val_loader, num_classes=num_classes, device=device)


# ----------------------------
# 6. Lancement
# ----------------------------
if __name__ == "__main__":
    train()

