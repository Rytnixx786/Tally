"""Setup A: TF-IDF + logistic regression. The baseline everyone skips."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

MODEL_NAME = "tfidf-word12-char35-logreg"


def build_pipeline(c: float, seed: int = 0) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ]
    )
    clf = LogisticRegression(C=c, max_iter=3000, random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class APredictions:
    labels: list[str]
    confidence: np.ndarray  # max class probability
    seconds_per_ticket: float


@dataclass
class AFit:
    model: Pipeline
    c: float
    tune_accuracy_by_c: dict[float, float]


def fit_select(dev: pd.DataFrame, tune: pd.DataFrame, c_grid: list[float], seed: int = 0) -> AFit:
    """Train on dev, choose C on tune. The test split is never seen here."""
    scores: dict[float, float] = {}
    best: tuple[float, Pipeline] | None = None
    for c in c_grid:
        m = build_pipeline(c, seed).fit(dev["text"], dev["intent"])
        acc = float(np.mean(m.predict(tune["text"]) == tune["intent"].to_numpy()))
        scores[c] = acc
        if best is None or acc > scores[best[0]]:
            best = (c, m)
    assert best is not None
    return AFit(model=best[1], c=best[0], tune_accuracy_by_c=scores)


def predict(model: Pipeline, df: pd.DataFrame) -> APredictions:
    t0 = time.perf_counter()
    proba = model.predict_proba(df["text"])
    elapsed = time.perf_counter() - t0
    classes = model.classes_
    idx = proba.argmax(axis=1)
    return APredictions(
        labels=[str(classes[i]) for i in idx],
        confidence=proba.max(axis=1),
        seconds_per_ticket=elapsed / max(len(df), 1),
    )


def learning_curve(
    dev: pd.DataFrame, eval_df: pd.DataFrame, per_intent: list[int], c: float, seed: int = 0
) -> list[dict[str, float]]:
    """Accuracy when training on k labelled examples per intent (0 = all of dev)."""
    rows = []
    for k in per_intent:
        train = (
            dev
            if k == 0
            else dev.groupby("intent", group_keys=False).sample(n=k, random_state=seed, replace=False)
            if dev.groupby("intent").size().min() >= k
            else dev
        )
        m = build_pipeline(c, seed).fit(train["text"], train["intent"])
        acc = float(np.mean(m.predict(eval_df["text"]) == eval_df["intent"].to_numpy()))
        rows.append({"per_intent": float(k), "n_train": float(len(train)), "accuracy": acc})
    return rows
