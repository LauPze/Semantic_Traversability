import os
import random
from glob import glob

# Définir les chemins
base_dir = "RUGD"
img_dir = os.path.join(base_dir, "images")
ann_dir = os.path.join(base_dir, "annotations")

# Récupérer tous les fichiers images (avec recherche dans les sous-dossiers)
image_paths = sorted(glob(os.path.join(img_dir, "**/*.png"), recursive=True))

# Extraire les noms relatifs pour le matching avec les annotations
rel_images = [os.path.relpath(p, img_dir) for p in image_paths]

# Shuffle pour split aléatoire (en gardant les paires image/annotation alignées)
random.seed(42)
combined = list(zip(image_paths, rel_images))
random.shuffle(combined)
image_paths[:], rel_images[:] = zip(*combined)

# Split 80% train / 20% val
split_idx = int(len(image_paths) * 0.8)
train_images = rel_images[:split_idx]
val_images = rel_images[split_idx:]

# Fichiers de sortie
train_txt = os.path.join(base_dir, "train.txt")
val_txt = os.path.join(base_dir, "val.txt")

def write_split_file(file_list, out_path):
    with open(out_path, 'w') as f:
        for rel_path in file_list:
            img_path = os.path.join(img_dir, rel_path)
            ann_path = os.path.join(ann_dir, rel_path)
            
            # Vérifier que l'annotation existe
            if not os.path.exists(ann_path):
                print(f"Attention: annotation manquante pour {img_path}")
                continue
                
            f.write(f"{img_path} {ann_path}\n")

write_split_file(train_images, train_txt)
write_split_file(val_images, val_txt)

print(f"Fichiers de split créés : {train_txt}, {val_txt}")
print(f"Total images: {len(rel_images)}")
print(f"Train: {len(train_images)}")
print(f"Val: {len(val_images)}")
