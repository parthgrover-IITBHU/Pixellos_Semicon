
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Arch


def load_input(path):
    arr = np.load(path)

    # Expected source format is a single-channel image stored as
    # (1,H,W). Also accept (H,W) for convenience.
    if arr.ndim == 2:
        arr = arr[None, ...]

    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(
            f"{path}: expected shape (1,H,W) or (H,W), got {arr.shape}"
        )

    return torch.from_numpy(arr).float().unsqueeze(0)


def save_output(tensor, path):
    img = tensor.squeeze().detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0)

    plt.imsave(path, img, cmap="gray", vmin=0.0, vmax=1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Run the trained image-restoration model on .npy test images."
    )
    parser.add_argument(
        "test_dir",
        help="Directory containing test .npy images"
    )
    parser.add_argument(
        "output_dir",
        help="Directory where restored images will be written"
    )
    parser.add_argument(
        "--weights",
        default="best_model.pt",
        help="Path to trained model state_dict (.pt)"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=32
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use the notebook's flip-based TTA (rotations disabled)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = Arch(args.channels).to(device)

    state = torch.load(
        args.weights,
        map_location=device
    )
    model.load_state_dict(state)
    model.eval()

    files = sorted(
        f for f in os.listdir(args.test_dir)
        if f.endswith(".npy")
    )

    if not files:
        raise RuntimeError(f"No .npy files found in: {args.test_dir}")

    with torch.no_grad():
        for idx, filename in enumerate(files, 1):
            x = load_input(os.path.join(args.test_dir, filename)).to(device)

            pred = model(x)

            stem = os.path.splitext(filename)[0]
            output_path = os.path.join(
                args.output_dir,
                stem + ".png"
            )
            save_output(pred, output_path)

            print(f"[{idx}/{len(files)}] {filename} -> {output_path}")


if __name__ == "__main__":
    main()
