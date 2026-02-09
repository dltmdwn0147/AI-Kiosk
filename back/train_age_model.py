import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image


class AgeFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, mode: str = "regression", max_decade: int = None):
        self.root = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.samples = []
        ages = []
        for age_dir in sorted(self.root.iterdir()):
            if not age_dir.is_dir():
                continue
            try:
                age = int(age_dir.name)
            except ValueError:
                continue
            ages.append(age)
            for img_path in age_dir.rglob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((img_path, age))
        if not self.samples:
            raise RuntimeError(f"No images found in {root_dir}")
        if self.mode == "decade":
            max_age = max(ages) if ages else 0
            self.max_decade = max_decade if max_decade is not None else (max_age // 10) * 10
            self.num_classes = self.max_decade // 10 + 1
        else:
            self.max_decade = None
            self.num_classes = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.mode == "decade":
            label = min((age // 10), self.num_classes - 1)
            return img, torch.tensor(label, dtype=torch.long)
        return img, torch.tensor(age, dtype=torch.float32)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int = 1):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device, loss_fn, mode: str):
    model.train()
    total_loss = 0.0
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        if mode == "regression":
            pred = pred.squeeze(1)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        if step % 50 == 0:
            print(f"  - train step {step}/{len(loader)} loss={loss.item():.4f}")
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, mode: str):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        if mode == "regression":
            pred = pred.squeeze(1)
        loss = loss_fn(pred, y)
        if mode == "regression":
            mae = torch.mean(torch.abs(pred - y))
        else:
            mae = (pred.argmax(dim=1) == y).float().mean()
        total_loss += loss.item() * x.size(0)
        total_mae += mae.item() * x.size(0)
        if step % 50 == 0:
            print(f"  - val step {step}/{len(loader)} loss={loss.item():.4f}")
    n = len(loader.dataset)
    return total_loss / n, total_mae / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="back/face_age", help="dataset root")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="back/age_model.pt")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (macOS permission issue -> 0 권장)")
    parser.add_argument("--mode", choices=["regression", "decade"], default="regression")
    parser.add_argument("--max-decade", type=int, default=None, help="최대 연령대(예: 80이면 0~80대)")
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"✅ device: {device}")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_ds = AgeFolderDataset(args.data_dir, transform=train_tf, mode=args.mode, max_decade=args.max_decade)
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    # use val transforms
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=full_ds.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss() if args.mode == "regression" else nn.CrossEntropyLoss()

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn, args.mode)
        val_loss, val_mae = evaluate(model, val_loader, device, loss_fn, args.mode)
        metric_label = "val_mae" if args.mode == "regression" else "val_acc"
        print(f"[{epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} {metric_label}={val_mae:.4f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"model": model.state_dict(), "mae": best_mae}, args.output)
            print(f"✅ saved: {args.output}")


if __name__ == "__main__":
    main()
