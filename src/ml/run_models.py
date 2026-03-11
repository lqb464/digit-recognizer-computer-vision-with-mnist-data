from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .advanced_models import run_advanced_models
from .base_models import run_base_models


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


def _load_train_csv(train_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Thiếu dependency `pandas`. Cài bằng: pip install pandas") from e

    df = pd.read_csv(train_csv)

    if "label" in df.columns:
        y = df["label"].to_numpy()
        X = df.drop(columns=["label"]).to_numpy()
    else:
        # fallback: assume first column is label (khi dataset không đặt tên cột)
        y = df.iloc[:, 0].to_numpy()
        X = df.iloc[:, 1:].to_numpy()

    return X, y


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    try:
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Thiếu dependency `scikit-learn`. Cài bằng: pip install scikit-learn"
        ) from e

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DatasetSplit(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)


def _maybe_scale_pixels(X_train: np.ndarray, X_val: np.ndarray, *, scale: bool):
    if not scale:
        return X_train, X_val

    # MNIST pixels are 0..255; scale to 0..1 helps some models converge faster.
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    return X_train, X_val


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chạy nhóm ML models trên MNIST train.csv (Kaggle Digit Recognizer)."
    )
    p.add_argument(
        "--mode",
        choices=["base", "advanced"],
        default="base",
        help="Chọn nhóm model để chạy.",
    )
    p.add_argument(
        "--train-csv",
        type=Path,
        default=Path("dataset/train.csv"),
        help="Đường dẫn tới train.csv (có cột label).",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Tỉ lệ validation split.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--scale",
        action="store_true",
        help="Scale pixels về 0..1 trước khi train.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Thư mục để ghi output. Mặc định: output/ml/<mode>/<timestamp>.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Tên run (để đặt subfolder). Mặc định dùng timestamp.",
    )
    return p.parse_args(argv)


def _default_outdir(mode: str, run_name: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or ts
    return Path("output") / "ml" / mode / name


def _write_results(outdir: Path, *, scores: dict[str, float], meta: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {"meta": meta, "scores": scores}
    (outdir / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"mode: {meta.get('mode')}",
        f"train_csv: {meta.get('train_csv')}",
        f"test_size: {meta.get('test_size')}",
        f"seed: {meta.get('seed')}",
        f"scale: {meta.get('scale')}",
        "",
        "scores:",
    ]
    for k, v in sorted(scores.items(), key=lambda kv: kv[0]):
        lines.append(f"- {k}: {v:.6f}")
    (outdir / "scores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    X, y = _load_train_csv(args.train_csv)
    split = _train_val_split(X, y, test_size=args.test_size, random_state=args.seed)

    X_train, X_val = _maybe_scale_pixels(split.X_train, split.X_val, scale=args.scale)

    if args.mode == "base":
        scores = run_base_models(
            X_train, split.y_train, X_val, split.y_val, random_state=args.seed
        )
    else:
        scores = run_advanced_models(
            X_train, split.y_train, X_val, split.y_val, random_state=args.seed
        )

    outdir = args.outdir or _default_outdir(args.mode, args.run_name)
    meta = {
        "mode": args.mode,
        "train_csv": str(args.train_csv),
        "test_size": args.test_size,
        "seed": args.seed,
        "scale": bool(args.scale),
    }
    _write_results(outdir, scores=scores, meta=meta)
    print(f"Wrote results to: {outdir}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

