#!/usr/bin/env python3
"""Generate DecoyMNIST PNG folders from MakeMNIST decoy arrays.

Pipeline:
1) Run MakeMNIST DecoyMNIST data generator (00_make_data.py).
2) Read generated .npy arrays (train_x_decoy.npy / test_x_decoy.npy).
3) Export PNGs to:
     <output-root>/train/<digit>/*.png
     <output-root>/test/<digit>/*.png

Default output root is compatible with existing repro runners:
  repro_runs/third_party/MakeMNIST/data/DecoyMNIST_png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST


def _default_repo_root() -> Path:
    here = Path(__file__).resolve()
    marker = Path("repro_runs") / "third_party" / "MakeMNIST" / "mnist" / "DecoyMNIST" / "00_make_data.py"
    for candidate in here.parents:
        if (candidate / marker).exists():
            return candidate
    return here.parent


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a single 2D image array to uint8 [0,255]."""
    x = np.asarray(img, dtype=np.float32)

    # Expected from 00_make_data.py: values in [-1, 1].
    if x.min() >= -1.01 and x.max() <= 1.01:
        x = (x + 1.0) * 0.5
    # Accept [0, 1] too.
    elif x.min() >= -1e-6 and x.max() <= 1.000001:
        pass
    # Fallback: robust min-max normalization.
    else:
        mn, mx = float(x.min()), float(x.max())
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x, dtype=np.float32)

    x = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    return x


def _export_split_pngs(
    split_name: str,
    x_path: Path,
    labels: np.ndarray,
    out_root: Path,
    overwrite: bool,
) -> int:
    x = np.load(x_path)
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Unexpected array shape in {x_path}: {x.shape}; expected (N,1,H,W)")

    n = x.shape[0]
    if labels.shape[0] != n:
        raise ValueError(
            f"Label count mismatch for {split_name}: array has {n}, labels have {labels.shape[0]}"
        )

    split_root = out_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx in range(n):
        y = int(labels[idx])
        cls_dir = split_root / str(y)
        cls_dir.mkdir(parents=True, exist_ok=True)

        out_path = cls_dir / f"{idx:06d}.png"
        if out_path.exists() and not overwrite:
            continue

        img_u8 = _to_uint8(x[idx, 0])
        Image.fromarray(img_u8, mode="L").save(out_path)
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DecoyMNIST_png by running MakeMNIST decoy generator.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="Repository root (default: auto-detected by searching parent directories for repro_runs/third_party/MakeMNIST).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Where to write PNG folders. Default: "
            "<repo-root>/repro_runs/third_party/MakeMNIST/data/DecoyMNIST_png"
        ),
    )
    parser.add_argument(
        "--skip-make-data",
        action="store_true",
        help="Skip running 00_make_data.py and use existing .npy arrays.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    makemnist_root = repo_root / "repro_runs" / "third_party" / "MakeMNIST"
    make_data_script = makemnist_root / "mnist" / "DecoyMNIST" / "00_make_data.py"
    data_root = makemnist_root / "data"
    color_root = data_root / "ColorMNIST"
    output_root = args.output_root.resolve() if args.output_root else (data_root / "DecoyMNIST_png")

    train_x_path = color_root / "train_x_decoy.npy"
    test_x_path = color_root / "test_x_decoy.npy"

    if not make_data_script.exists():
        raise FileNotFoundError(f"Missing generator script: {make_data_script}")

    color_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_make_data:
        cmd = [sys.executable, str(make_data_script)]
        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(make_data_script.parent))

    if not train_x_path.exists() or not test_x_path.exists():
        raise FileNotFoundError(
            "Decoy arrays not found after generation. Expected:\n"
            f"  {train_x_path}\n"
            f"  {test_x_path}"
        )

    # Labels come from canonical MNIST labels in the same MakeMNIST data root.
    train_labels = np.asarray(MNIST(root=str(data_root), train=True, download=True).targets)
    test_labels = np.asarray(MNIST(root=str(data_root), train=False, download=True).targets)

    output_root.mkdir(parents=True, exist_ok=True)
    train_written = _export_split_pngs("train", train_x_path, train_labels, output_root, args.overwrite)
    test_written = _export_split_pngs("test", test_x_path, test_labels, output_root, args.overwrite)

    print("[DONE] DecoyMNIST PNG export complete")
    print(f"  output_root: {output_root}")
    print(f"  train_written: {train_written}")
    print(f"  test_written: {test_written}")


if __name__ == "__main__":
    main()
