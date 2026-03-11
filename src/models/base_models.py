"""Một vài baseline models nhẹ, dùng để benchmark nhanh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class BaseEvalResult:
    scores: dict[str, float]
    trained_models: dict[str, object]



def train_eval_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    random_state: int = 42,
) -> BaseEvalResult:
    models = {
        "logreg": LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            n_jobs=None,
        ),
        "gnb": GaussianNB(),
        "dt": DecisionTreeClassifier(random_state=random_state, max_depth=20),
    }

    scores: dict[str, float] = {}
    trained_models: dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = float(accuracy_score(y_val, y_pred))
        scores[name] = acc
        trained_models[name] = model
        print(f"{name} accuracy: {acc:.5f}")

    return BaseEvalResult(scores=scores, trained_models=trained_models)



def train_base_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
):
    if model_name == "logreg":
        model = LogisticRegression(random_state=random_state, max_iter=1000, n_jobs=None)
    elif model_name == "gnb":
        model = GaussianNB()
    elif model_name == "dt":
        model = DecisionTreeClassifier(random_state=random_state, max_depth=20)
    else:
        raise ValueError(f"Unsupported base model: {model_name}")

    model.fit(X_train, y_train)
    return model