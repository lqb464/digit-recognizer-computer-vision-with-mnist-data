from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class MlpConfig:
    hidden_layer_sizes: tuple[int, ...] = (256, 128)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 1e-4
    batch_size: int | str = 256
    learning_rate_init: float = 1e-3
    max_iter: int = 50
    early_stopping: bool = True
    validation_fraction: float = 0.1


@dataclass(frozen=True)
class MlpEvalResult:
    scores: dict[str, float]
    model: MLPClassifier



def build_mlp_classifier(*, random_state: int = 42, cfg: MlpConfig | None = None) -> MLPClassifier:
    cfg = cfg or MlpConfig()
    return MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        activation=cfg.activation,
        solver=cfg.solver,
        alpha=cfg.alpha,
        batch_size=cfg.batch_size,
        learning_rate_init=cfg.learning_rate_init,
        max_iter=cfg.max_iter,
        early_stopping=cfg.early_stopping,
        validation_fraction=cfg.validation_fraction,
        random_state=random_state,
        verbose=False,
    )



def train_eval_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    random_state: int = 42,
    cfg: MlpConfig | None = None,
) -> MlpEvalResult:
    cfg = cfg or MlpConfig()
    model = build_mlp_classifier(random_state=random_state, cfg=cfg)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    print(f"mlp accuracy: {acc:.5f}")
    return MlpEvalResult(scores={"mlp": acc}, model=model)



def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
    cfg: MlpConfig | None = None,
) -> MLPClassifier:
    cfg = cfg or MlpConfig()
    model = build_mlp_classifier(random_state=random_state, cfg=cfg)
    model.fit(X_train, y_train)
    return model