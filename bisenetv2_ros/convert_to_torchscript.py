import torch
from bisenetv2_ros.bisenetv2 import BiSeNetV2  # adapte l'import à ton chemin
import torchvision.transforms as T
import numpy as np

# === Paramètres ===
n_classes = 20
model_path = '/home/Laura/src/bisenetv2_ros/models/my_model.pth'
output_path = '/home/Laura/src/bisenetv2_ros/models/bisenetv2_scripted.pt'

# === Charger le modèle ===
model = BiSeNetV2(n_classes=n_classes)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# === Dummy input (même taille que dans ton nœud) ===
dummy_input = torch.randn(1, 3, 512, 256)  # C, H, W : change si besoin

# === Tracer le modèle avec torch.jit ===
scripted_model = torch.jit.trace(model, dummy_input)

# === Sauvegarde ===
scripted_model.save(output_path)

print(f"Modèle TorchScript exporté : {output_path}")
