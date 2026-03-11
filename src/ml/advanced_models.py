"""Các model ML nặng hơn (ensemble/boosting/SVC/stacking...)."""

from __future__ import annotations

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def run_advanced_models(X_train, y_train, X_test, y_test, *, random_state: int = 42):
    """Train/eval một vài model nặng hơn.

    Returns:
        dict[str, float]: accuracy theo từng model.
    """
    models = {
        "rf": RandomForestClassifier(
            random_state=random_state,
            n_estimators=300,
            n_jobs=-1,
        ),
        "svc_rbf": SVC(kernel="rbf"),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "gb": GradientBoostingClassifier(random_state=random_state),
        "ada": AdaBoostClassifier(random_state=random_state),
        "extra_trees": ExtraTreesClassifier(
            random_state=random_state,
            n_estimators=500,
            n_jobs=-1,
        ),
        "bagging_dt": BaggingClassifier(
            random_state=random_state,
            n_estimators=200,
            n_jobs=-1,
        ),
    }

    scores: dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores[name] = acc
        print(f"{name} accuracy: {acc:.5f}")

    return scores

