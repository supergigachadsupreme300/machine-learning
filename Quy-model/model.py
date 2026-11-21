"""Model definitions for Quy-model (landmark-based)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(name: str) -> Pipeline:
    name = name.lower()
    scaler = StandardScaler()
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    elif name == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=42)
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=42)
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    elif name == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(
            f"Unknown model name: {name}. "
            f"Choose from random_forest, svm, knn, gradient_boosting, mlp, logistic."
        )
    return Pipeline([("scaler", scaler), ("clf", clf)])
