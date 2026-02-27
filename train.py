import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm


def build_model(model_name='vit_base_patch16_224', num_classes=2, pretrained=True, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = None
    if hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    model.to(device)
    return model, device


def train(data_dir, out_path='model_finetuned.pth', epochs=5, batch_size=16, lr=3e-4):
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    # Validate dataset structure early with a clear error message
    required = [train_dir, train_dir / 'real', train_dir / 'fake', val_dir, val_dir / 'real', val_dir / 'fake']
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print('\nERROR: Dataset folders are missing. Expected the following directories:')
        for p in required:
            print(' -', p)
        print('\nPlease create the directories and place images inside. You can generate sample placeholders with `python create_sample_data.py --out data`')
        raise SystemExit(1)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model, device = build_model()

    # freeze backbone except head
    for name, p in model.named_parameters():
        if 'head' not in name and 'fc' not in name:
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            running += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = running / total if total>0 else 0.0

        # eval
        model.eval()
        running = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                running += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = running / total if total>0 else 0.0
        print(f'Epoch {epoch+1}/{epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state': model.state_dict()}, out_path)
            print('Saved best model to', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data directory with train/val subfolders')
    parser.add_argument('--out', default='model_finetuned.pth')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    train(args.data, out_path=args.out, epochs=args.epochs, batch_size=args.batch, lr=args.lr)


if __name__ == '__main__':
    main()
