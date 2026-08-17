
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from model import Arch
from losses import RestorationLoss


def load_npy_dir(directory):
    files = sorted(
        f for f in os.listdir(directory)
        if f.endswith(".npy")
    )
    if not files:
        raise RuntimeError(f"No .npy files found in: {directory}")

    return np.array([
        np.load(os.path.join(directory, f))
        for f in files
    ])


def train_val_split(x, y, val_fraction=0.2, seed=42):
    assert len(x) == len(y)
    n = len(x)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_val = int(n * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def make_loader(x, y, batch_size, shuffle):
    x_tensor = torch.from_numpy(x).unsqueeze(0).permute(1, 0, 2, 3)
    y_tensor = torch.from_numpy(y).unsqueeze(0).permute(1, 0, 2, 3)
    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle
    )


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    psnr_total = 0.0
    n_batches = 0

    with torch.no_grad():
        for lr_img, gt_img in loader:
            lr_img = lr_img.to(device)
            gt_img = gt_img.to(device)

            pred = model(lr_img)

            if pred.shape[-2:] != gt_img.shape[-2:]:
                pred = F.interpolate(
                    pred, size=gt_img.shape[-2:],
                    mode="bilinear", align_corners=False
                )

            loss, _ = criterion(pred, gt_img)
            running_loss += loss.item()

            mse = F.mse_loss(pred, gt_img)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
            psnr_total += psnr.item()
            n_batches += 1

    return running_loss / n_batches, psnr_total / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", default="best_model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    x_all = load_npy_dir(args.lr_dir)
    y_all = load_npy_dir(args.gt_dir)

    x_train, y_train, x_val, y_val = train_val_split(
        x_all, y_all, args.val_fraction, args.seed
    )

    train_loader = make_loader(
        x_train, y_train, args.batch_size, shuffle=True
    )
    val_loader = make_loader(
        x_val, y_val, args.batch_size, shuffle=False
    )

    model = Arch(args.channels).to(device)
    criterion = RestorationLoss(w_edge=1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    best_val_psnr = -float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start = time.time()

        for lr_img, gt_img in train_loader:
            lr_img = lr_img.to(device)
            gt_img = gt_img.to(device)

            optimizer.zero_grad()
            pred = model(lr_img)

            if pred.shape[-2:] != gt_img.shape[-2:]:
                pred = F.interpolate(
                    pred, size=gt_img.shape[-2:],
                    mode="bilinear", align_corners=False
                )

            loss, _ = criterion(pred, gt_img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_psnr = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}: "
            f"avg_loss={avg_train_loss:.5f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}, "
            f"val_loss={val_loss:.5f}, "
            f"val_PSNR={val_psnr:.4f}, "
            f"time={time.time() - start:.1f}s"
        )

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), args.output)
            print(
                f"  -> new best val PSNR: {best_val_psnr:.2f}, "
                f"saved {args.output}"
            )

        scheduler.step()


if __name__ == "__main__":
    main()
