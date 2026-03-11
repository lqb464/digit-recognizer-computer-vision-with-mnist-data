from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    metrics_json: Path
    metrics_txt: Path
    meta_json: Path
    leaderboard_csv: Path
    model_path: Path
    submission_path: Path


@dataclass(frozen=True)
class ModelSelection:
    best_model_name: str
    best_score: float
    all_scores: dict[str, float]


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Không tìm thấy {label}: {path}")


def load_train_csv(train_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Thiếu dependency `pandas`. Cài bằng: pip install pandas") from e

    ensure_exists(train_csv, "train.csv")
    df = pd.read_csv(train_csv)

    if "label" in df.columns:
        y = df["label"].to_numpy()
        X = df.drop(columns=["label"]).to_numpy()
    else:
        y = df.iloc[:, 0].to_numpy()
        X = df.iloc[:, 1:].to_numpy()

    return X, y


def load_test_csv(test_csv: Path) -> np.ndarray:
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Thiếu dependency `pandas`. Cài bằng: pip install pandas") from e

    ensure_exists(test_csv, "test.csv")
    df = pd.read_csv(test_csv)

    if "label" in df.columns:
        df = df.drop(columns=["label"])
    return df.to_numpy()


def train_val_split(
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


def scale_01(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_train = X_train.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    return X_train, X_val


def scale_single_01(X: np.ndarray) -> np.ndarray:
    return X.astype(np.float32) / 255.0


def parse_hidden_layers(s: str) -> tuple[int, ...]:
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
    p = argparse.ArgumentParser(
        description="Train/eval baseline models và MLP cho Kaggle Digit Recognizer."
    )
    p.add_argument(
        "--model",
        choices=["base", "mlp", "all"],
        default="all",
        help="Chọn nhóm model cần chạy ở bước selection.",
    )
    p.add_argument(
        "--predict-only",
        action="store_true",
        help="Chỉ inference bằng model đã lưu từ --model-path.",
    )
    p.add_argument(
        "--make-submission",
        action="store_true",
        help="Ghi submission.csv từ final model.",
    )
    p.add_argument(
        "--train-csv",
        type=Path,
        default=Path("dataset/train.csv"),
        help="Đường dẫn train.csv.",
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=Path("dataset/test.csv"),
        help="Đường dẫn test.csv.",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Đường dẫn model .joblib để load khi dùng --predict-only.",
    )
    p.add_argument(
        "--selection-dir",
        type=Path,
        default=None,
        help="Thư mục output của bước 1, dùng để đọc best model khi chạy --fit-full-train.",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Tỉ lệ validation split.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--hidden-layers",
        type=parse_hidden_layers,
        default=(256, 128),
        help='Ví dụ: "256,128" hoặc "512,256,128".',
    )
    p.add_argument("--max-iter", type=int, default=50, help="Số iteration tối đa cho MLP.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size cho MLP.")
    p.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization cho MLP.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate init cho MLP.")
    p.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Tắt early stopping của MLP.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Thư mục output. Mặc định: output/<model_name>/<run_name|timestamp>/.",
    )
    p.add_argument(
        "--submission-path",
        type=Path,
        default=None,
        help="Đường dẫn submission.csv. Mặc định: <outdir>/submission.csv.",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Tên run. Mặc định dùng timestamp.",
    )
    p.add_argument(
        "--fit-full-train",
        action="store_true",
        help="Train lại duy nhất best model trên toàn bộ train.csv.",
    )

    args = p.parse_args(argv)

    if args.predict_only and args.model_path is None:
        p.error("--predict-only cần --model-path.")
    if args.fit_full_train and args.selection_dir is None:
        p.error("--fit-full-train cần --selection-dir.")

    return args


def default_outdir(run_name: str | None, model_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name or ts
    return Path("output") / model_name / name


def prepare_output_paths(
    *,
    model_name: str,
    run_name: str | None,
    outdir: Path | None,
    submission_path: Path | None,
) -> OutputPaths:
    root = outdir or default_outdir(run_name, model_name)
    root.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        root=root,
        metrics_json=root / "metrics.json",
        metrics_txt=root / "metrics.txt",
        meta_json=root / "meta.json",
        leaderboard_csv=root / "leaderboard.csv",
        model_path=root / "model.joblib",
        submission_path=submission_path or (root / "submission.csv"),
    )


def select_best_model(scores: dict[str, float]) -> ModelSelection:
    if not scores:
        raise SystemExit("Không có score nào để chọn best model.")
    best_model_name = max(scores, key=scores.get)
    return ModelSelection(
        best_model_name=best_model_name,
        best_score=float(scores[best_model_name]),
        all_scores={k: float(v) for k, v in scores.items()},
    )


def write_results(
    output: OutputPaths,
    *,
    scores: dict[str, float],
    meta: dict[str, Any],
    best_model_name: str,
) -> None:
    best_score = float(scores[best_model_name]) if best_model_name in scores else None

    payload = {
        "meta": meta,
        "scores": {k: float(v) for k, v in scores.items()},
        "best_model_name": best_model_name,
        "best_model_score": best_score,
    }
    output.metrics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output.meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = ["meta:"]
    for k, v in meta.items():
        lines.append(f"- {k}: {v}")

    if scores:
        lines += ["", "scores:"]
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for idx, (name, score) in enumerate(sorted_scores, start=1):
            marker = " <= best" if name == best_model_name else ""
            lines.append(f"{idx}. {name}: {score:.6f}{marker}")
    else:
        lines += ["", f"best_model_name: {best_model_name}"]
        if best_score is None:
            lines.append("scores: <not available in this phase>")

    output.metrics_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Thiếu dependency `pandas`. Cài bằng: pip install pandas") from e

    if scores:
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        leaderboard_df = pd.DataFrame(
            [
                {"rank": idx, "model": name, "accuracy": float(score)}
                for idx, (name, score) in enumerate(sorted_scores, start=1)
            ]
        )
    else:
        leaderboard_df = pd.DataFrame(
            [{"rank": 1, "model": best_model_name, "accuracy": np.nan}]
        )
    leaderboard_df.to_csv(output.leaderboard_csv, index=False)


def save_model(model, model_path: Path) -> None:
    try:
        import joblib
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Không import được `joblib` (thường đi kèm scikit-learn). Thử: pip install joblib"
        ) from e
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path):
    try:
        import joblib
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Không import được `joblib` (thường đi kèm scikit-learn). Thử: pip install joblib"
        ) from e
    ensure_exists(model_path, "model file")
    return joblib.load(model_path)


def write_submission(submission_path: Path, y_pred: np.ndarray) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit("Thiếu dependency `pandas`. Cài bằng: pip install pandas") from e

    df = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(y_pred) + 1, dtype=np.int64),
            "Label": y_pred.astype(np.int64),
        }
    )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(submission_path, index=False)
