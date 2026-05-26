"""
Entrena MobileNetV3-small sobre los crops recolectados con recopilar_datos.py.
Exporta dorsal_classifier.onnx + dorsal_classes.json para usar en test_ocr.py.

Uso: python entrenar_dorsal.py
"""
import torch, torch.nn as nn, torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import numpy as np, json, os

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'training_data'
EPOCHS   = 30
BATCH    = 32
IMG_SIZE = 96

print(f"Entrenando en: {DEVICE}")

if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    print(f"ERROR: '{DATA_DIR}/' vacío. Ejecutá primero: python recopilar_datos.py")
    exit(1)

transform_train = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.2),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transform_val = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_ds = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform_train)
classes = full_ds.classes
num_classes = len(classes)

n_val = max(1, int(len(full_ds) * 0.15))
train_idx, val_idx = random_split(
    range(len(full_ds)), [len(full_ds) - n_val, n_val]
)

train_ds = torch.utils.data.Subset(full_ds, train_idx.indices)
val_ds_base = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform_val)
val_ds = torch.utils.data.Subset(val_ds_base, val_idx.indices)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

print(f"Clases: {classes}")
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

# MobileNetV3-small con ImageNet pretraining
net = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
net = net.to(DEVICE)

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    net.train()
    total_loss = correct = total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = net(imgs)
        loss = criterion(out, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    scheduler.step()

    net.eval()
    vc = vt = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            vc += (net(imgs).argmax(1) == labels).sum().item()
            vt += labels.size(0)
    val_acc = vc / vt if vt > 0 else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), 'dorsal_best.pt')

    print(f"Epoch {epoch:2d}/{EPOCHS}  "
          f"loss={total_loss/len(train_loader):.3f}  "
          f"train={100*correct/total:.1f}%  "
          f"val={100*val_acc:.1f}%")

# Exportar mejor modelo
net.load_state_dict(torch.load('dorsal_best.pt', weights_only=True))
net.eval()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
torch.onnx.export(net, dummy, 'dorsal_classifier.onnx',
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch'}}, opset_version=17)

with open('dorsal_classes.json', 'w') as f:
    json.dump(classes, f)

print(f"\nExportado → dorsal_classifier.onnx")
print(f"Clases detectadas: {classes}")
print(f"Mejor val accuracy: {100*best_val_acc:.1f}%")
print("Siguiente: python test_ocr.py")
