from __future__ import annotations

import json
import sys
from pathlib import Path

from utils import (
    load_model,
    load_test_csv,
    load_train_csv,
    parse_args,
    prepare_output_paths,
    save_model,
    scale_01,
    scale_single_01,
    select_best_model,
    train_val_split,
    write_results,
    write_submission,
)
from models.base_models import train_base_model, train_eval_base_models
from models.mlp_model import MlpConfig, train_eval_mlp, train_mlp


def _load_best_model_name(selection_dir: str | Path) -> str:
    selection_dir = Path(selection_dir)
    metrics_path = selection_dir / "metrics.json"
    meta_path = selection_dir / "meta.json"

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        best_model_name = metrics.get("best_model_name")
        if best_model_name:
            return str(best_model_name)

    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        best_model_name = meta.get("best_model_name")
        if best_model_name:
            return str(best_model_name)

    raise FileNotFoundError(
        f"Không tìm thấy best_model_name trong: {selection_dir}"
    )


def _build_mlp_cfg(args) -> MlpConfig:
    return MlpConfig(
        hidden_layer_sizes=tuple(args.hidden_layers),
        alpha=float(args.alpha),
        batch_size=int(args.batch_size),
        learning_rate_init=float(args.lr),
        max_iter=int(args.max_iter),
        early_stopping=not bool(args.no_early_stopping),
    )


def _run_selection(args, cfg) -> int:
    output = prepare_output_paths(
        model_name=args.model,
        run_name=args.run_name,
        outdir=args.outdir,
        submission_path=args.submission_path,
    )

    X, y = load_train_csv(args.train_csv)
    split = train_val_split(X, y, test_size=args.test_size, random_state=args.seed)
    X_train_scaled, X_val_scaled = scale_01(split.X_train, split.X_val)

    all_scores: dict[str, float] = {}

    if args.model in {"base", "all"}:
        base_result = train_eval_base_models(
            X_train_scaled,
            split.y_train,
            X_val_scaled,
            split.y_val,
            random_state=args.seed,
        )
        all_scores.update(base_result.scores)

    if args.model in {"mlp", "all"}:
        mlp_result = train_eval_mlp(
            X_train_scaled,
            split.y_train,
            X_val_scaled,
            split.y_val,
            random_state=args.seed,
            cfg=cfg,
        )
        all_scores.update(mlp_result.scores)

    selection = select_best_model(all_scores)

    meta = {
        "phase": "selection",
        "mode": args.model,
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "test_size": args.test_size,
        "seed": args.seed,
        "scale": "0..1",
        "fit_full_train": False,
        "make_submission": False,
        "best_model_name": selection.best_model_name,
        "hidden_layers": list(cfg.hidden_layer_sizes),
        "max_iter": cfg.max_iter,
        "batch_size": cfg.batch_size,
        "alpha": cfg.alpha,
        "lr": cfg.learning_rate_init,
        "early_stopping": cfg.early_stopping,
    }
    write_results(
        output,
        scores=all_scores,
        meta=meta,
        best_model_name=selection.best_model_name,
    )

    print(f"Best model: {selection.best_model_name} ({selection.best_score:.6f})")
    print(f"Wrote selection results to: {output.root}")
    return 0


def _run_final_train(args, cfg) -> int:
    if getattr(args, "selection_dir", None) is None:
        raise SystemExit("--fit-full-train cần --selection-dir để đọc best model từ bước 1.")

    best_model_name = _load_best_model_name(args.selection_dir)

    output = prepare_output_paths(
        model_name="final",
        run_name=args.run_name,
        outdir=args.outdir,
        submission_path=args.submission_path,
    )

    X, y = load_train_csv(args.train_csv)
    X_full_scaled = scale_single_01(X)

    if best_model_name == "mlp":
        final_model = train_mlp(X_full_scaled, y, random_state=args.seed, cfg=cfg)
    else:
        final_model = train_base_model(best_model_name, X_full_scaled, y, random_state=args.seed)

    save_model(final_model, output.model_path)

    meta = {
        "phase": "final_train",
        "mode": args.model,
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "seed": args.seed,
        "scale": "0..1",
        "fit_full_train": True,
        "make_submission": bool(args.make_submission),
        "selection_dir": str(args.selection_dir),
        "best_model_name": best_model_name,
        "hidden_layers": list(cfg.hidden_layer_sizes),
        "max_iter": cfg.max_iter,
        "batch_size": cfg.batch_size,
        "alpha": cfg.alpha,
        "lr": cfg.learning_rate_init,
        "early_stopping": cfg.early_stopping,
    }
    write_results(
        output,
        scores={},
        meta=meta,
        best_model_name=best_model_name,
    )

    print(f"Saved final best model ({best_model_name}) to: {output.model_path}")

    if args.make_submission:
        X_test_scaled = scale_single_01(load_test_csv(args.test_csv))
        y_pred = final_model.predict(X_test_scaled)
        write_submission(output.submission_path, y_pred)
        print(f"Wrote submission to: {output.submission_path}")

    print(f"Wrote final-train results to: {output.root}")
    return 0


def _run_predict_only(args) -> int:
    output = prepare_output_paths(
        model_name="predict",
        run_name=args.run_name,
        outdir=args.outdir,
        submission_path=args.submission_path,
    )
    if args.model_path is None:
        raise SystemExit("--predict-only cần --model-path.")

    model = load_model(args.model_path)
    X_test = scale_single_01(load_test_csv(args.test_csv))
    y_pred = model.predict(X_test)
    write_submission(output.submission_path, y_pred)
    print(f"Wrote submission to: {output.submission_path}")
    return 0



def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args(argv)

    if args.predict_only:
        return _run_predict_only(args)

    cfg = _build_mlp_cfg(args)

    if args.fit_full_train:
        return _run_final_train(args, cfg)

    return _run_selection(args, cfg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())