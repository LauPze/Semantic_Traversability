import os
import random
from glob import glob

# Définir les chemins
rel_path = "00003/pylon_camera_node/frame001163-1581624191_549.jpg"

base_dir = "/mnt/915a88da-21d5-43c2-bc82-1e57a4212262/Stage_Laura/RELLIS-3D"

#base_dir = "RELLIS-3D"
img_dir = os.path.join(base_dir, "images", "Rellis-3D")
ann_dir = os.path.join(base_dir, "annotations", "Rellis-3D")

# Récupérer tous les fichiers images (avec recherche dans les sous-dossiers)
image_paths = sorted(glob(os.path.join(img_dir, "*", "pylon_camera_node", "*.jpg"), recursive=True))

ann_rel_path = rel_path.replace("pylon_camera_node", "pylon_camera_node_label_color").replace(".jpg", ".png")
ann_path = os.path.join(ann_dir, ann_rel_path)

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
    count_written = 0
    count_missing = 0
    with open(out_path, 'w') as f:
        for rel_path in file_list:
            img_path = os.path.join(img_dir, rel_path)
            ann_rel_path = rel_path.replace("pylon_camera_node", "pylon_camera_node_label_color").replace(".jpg", ".png")
            ann_path = os.path.join(ann_dir, ann_rel_path)           
            
            # Vérifier que l'annotation existe
            if not os.path.exists(ann_path):
                #print(f"Attention: annotation manquante pour {img_path}")
                count_missing += 1
                continue
                
            f.write(f"{img_path} {ann_path}\n")
            count_written += 1
    print(f"Écrit {count_written} paires dans {out_path}, ignoré {count_missing} images sans annotation")
write_split_file(train_images, train_txt)
write_split_file(val_images, val_txt)

print(f"Fichiers de split créés : {train_txt}, {val_txt}")
print(f"Total images: {len(rel_images)}")
print(f"Train: {len(train_images)}")
print(f"Val: {len(val_images)}")
