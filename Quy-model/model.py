"""Model definitions for Quy-model (landmark-based)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def build_model(name: str):
    name = name.lower()
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=42)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=42)
    raise ValueError(f"Unknown model: {name}")
