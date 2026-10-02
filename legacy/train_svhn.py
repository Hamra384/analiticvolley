"""
Entrena un CNN pequeño en SVHN format-2 (dígitos individuales 0-9)
y exporta a ONNX para usarlo en test_ocr.py.

Uso: python train_svhn.py
Salida: digit_classifier.onnx  (~300 KB)
Tiempo: ~10-15 min en GTX 1660 Super
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
BATCH  = 256
print(f"Entrenando en: {DEVICE}")

# SVHN format-2: 32x32 color, etiquetas 1-10 donde 10=0
transform = transforms.Compose([
    transforms.Grayscale(),          # → 1 canal, igual que nuestros blobs CC
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

print("Descargando SVHN (solo la primera vez)...")
train_ds = torchvision.datasets.SVHN(root='./svhn_data', split='train',
                                     download=True, transform=transform)
test_ds  = torchvision.datasets.SVHN(root='./svhn_data', split='test',
                                     download=True, transform=transform)

# Corregir etiquetas: SVHN usa 10 para el dígito 0
train_ds.labels = train_ds.labels % 10
test_ds.labels  = test_ds.labels  % 10

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                                # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model     = DigitCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    scheduler.step()

    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            val_correct += (model(imgs).argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    print(f"Epoch {epoch:2d}/{EPOCHS}  "
          f"loss={total_loss/len(train_loader):.3f}  "
          f"train={100*correct/total:.1f}%  "
          f"val={100*val_correct/val_total:.1f}%")

# Exportar a ONNX
model.eval()
dummy = torch.randn(1, 1, 32, 32).to(DEVICE)
torch.onnx.export(
    model, dummy, 'digit_classifier.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=17,
)
print("\nExportado → digit_classifier.onnx")
print("Ahora podés correr test_ocr.py")
