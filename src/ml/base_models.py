"""Một vài ML baseline models nhẹ & chạy nhanh.

Các model nặng hơn được chuyển sang `advanced_models.py`.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def run_base_models(X_train, y_train, X_test, y_test, *, random_state: int = 42):
    """Train/eval nhanh vài baseline models.

    Returns:
        dict[str, float]: accuracy theo từng model.
    """
    models = {
        "logreg": LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            n_jobs=None,  # giữ tương thích cho nhiều phiên bản sklearn
        ),
        "gnb": GaussianNB(),
        "dt": DecisionTreeClassifier(random_state=random_state, max_depth=20),
    }

    scores: dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores[name] = acc
        print(f"{name} accuracy: {acc:.5f}")

    return scores

