"""Train a small CNN to classify pasta shapes (spaghetti/tagliatelle/fusilli/penne).

Run generate_pasta.py first to build data/pasta/<class>/*.png.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

CLASSES = ["spaghetti", "tagliatelle", "fusilli", "penne"]
DATA_DIR = "data/pasta"


class PastaDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR):
        self.samples = []
        for label, cls in enumerate(CLASSES):
            for path in sorted(glob.glob(os.path.join(data_dir, cls, "*.png"))):
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return tensor, label


class PastaCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
        )
        self.classifier = nn.Linear(16 * 16 * 16, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def run(epochs, lr, batch_size, seed=42):
    torch.manual_seed(seed)

    dataset = PastaDataset()
    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = PastaCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / n_train

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
        val_acc = correct / n_val

        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"final val_accuracy={val_acc:.4f}")
    return val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    run(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
