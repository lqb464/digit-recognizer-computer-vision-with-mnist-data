from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .mlp_model import MlpConfig, train_eval_mlp


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


def _scale_01(X_train: np.ndarray, X_val: np.ndarray):
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    return X_train, X_val


def _parse_hidden_layers(s: str) -> tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("hidden-layers không hợp lệ. Ví dụ: 256,128")
    try:
        layers = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("hidden-layers phải là danh sách số nguyên.") from e
    if any(n <= 0 for n in layers):
        raise argparse.ArgumentTypeError("hidden-layers phải > 0.")
    return layers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chạy MLP (neural network) cho MNIST train.csv.")
    p.add_argument(
        "--train-csv",
        type=Path,
        default=Path("dataset/train.csv"),
        help="Đường dẫn tới train.csv (có cột label).",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Tỉ lệ validation split.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--hidden-layers",
        type=_parse_hidden_layers,
        default=(256, 128),
        help='Ví dụ: "256,128" hoặc "512,256,128".',
    )
    p.add_argument("--max-iter", type=int, default=50, help="Số epoch/iteration tối đa.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    p.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate init.")
    p.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Tắt early stopping (mặc định đang bật).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Thư mục để ghi output. Mặc định: output/nn/<timestamp>.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Tên run (để đặt subfolder). Mặc định dùng timestamp.",
    )
    return p.parse_args(argv)


def _default_outdir(run_name: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or ts
    return Path("output") / "nn" / name


def _write_results(outdir: Path, *, scores: dict[str, float], meta: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "scores": scores}
    (outdir / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [f"{k}: {v}" for k, v in meta.items()]
    lines += ["", "scores:"]
    for k, v in sorted(scores.items(), key=lambda kv: kv[0]):
        lines.append(f"- {k}: {v:.6f}")
    (outdir / "scores.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    X, y = _load_train_csv(args.train_csv)
    split = _train_val_split(X, y, test_size=args.test_size, random_state=args.seed)
    X_train, X_val = _scale_01(split.X_train, split.X_val)

    cfg = MlpConfig(
        hidden_layer_sizes=tuple(args.hidden_layers),
        alpha=float(args.alpha),
        batch_size=int(args.batch_size),
        learning_rate_init=float(args.lr),
        max_iter=int(args.max_iter),
        early_stopping=not bool(args.no_early_stopping),
    )

    scores = train_eval_mlp(
        X_train, split.y_train, X_val, split.y_val, random_state=args.seed, cfg=cfg
    )

    outdir = args.outdir or _default_outdir(args.run_name)
    meta = {
        "model": "mlp",
        "train_csv": str(args.train_csv),
        "test_size": args.test_size,
        "seed": args.seed,
        "scale": "0..1",
        "hidden_layers": list(cfg.hidden_layer_sizes),
        "max_iter": cfg.max_iter,
        "batch_size": cfg.batch_size,
        "alpha": cfg.alpha,
        "lr": cfg.learning_rate_init,
        "early_stopping": cfg.early_stopping,
    }
    _write_results(outdir, scores=scores, meta=meta)
    print(f"Wrote results to: {outdir}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

