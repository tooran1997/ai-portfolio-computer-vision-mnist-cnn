#!/usr/bin/env python
import argparse, os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from utils import train_val_split

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainval = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)
    test = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    train_ds, val_ds = train_val_split(trainval, args.val_ratio, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    run_dir = Path("runs") / "mnist_cnn"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir.as_posix())

    best_val = 0.0
    best_path = Path("runs") / "best_mnist_cnn.pt"

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            acc = accuracy(out, yb)
            pbar.set_postfix(loss=loss.item(), acc=acc)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/acc", acc, global_step)
            global_step += 1

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += crit(out, yb).item() * xb.size(0)
                val_correct += (out.argmax(dim=1) == yb).sum().item()
                val_total += xb.size(0)
        val_acc = val_correct / val_total
        val_loss /= val_total
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

        print(f"Epoch {epoch}: val_acc={val_acc:.4f} best={best_val:.4f}")

    # test
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            test_correct += (out.argmax(dim=1) == yb).sum().item()
            test_total += xb.size(0)
    test_acc = test_correct / test_total
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
