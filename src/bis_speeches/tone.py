"""Supervised sentence-level stance model and mixed-signal speech aggregation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import FeatureUnion


LABELS = ["dovish", "hawkish", "irrelevant", "neutral"]
ALPHAS = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
RANDOM_STATE = 42

POLICY_ANCHOR_RE = re.compile(
    r"\b(?:"
    r"inflation(?:ary)?|price\s+stability|price\s+pressures?|"
    r"monetary\s+policy|interest\s+rates?|policy\s+rates?|bank\s+rate|cash\s+rate|"
    r"repo\s+rate|refinancing\s+rate|deposit\s+(?:rate|facility)|"
    r"tighten(?:ing|ed)?|eas(?:e|ing|ed)|accommodat(?:ive|ion)|restrictive|"
    r"neutral\s+rate|output\s+gap|labou?r\s+market|wage\s+(?:growth|pressures?)|"
    r"demand\s+pressures?|forward\s+guidance|quantitative\s+(?:easing|tightening)|"
    r"asset\s+purchases?|balance\s+sheet|reserve\s+requirements?"
    r")\b",
    flags=re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])|[\r\n]+")
CONTRAST_SPLIT_RE = re.compile(
    r"\s+(?:but|however|although|though|whereas|while|yet)\s+|;\s+",
    flags=re.IGNORECASE,
)


@dataclass
class ToneModel:
    vectorizer: FeatureUnion
    classifier: CalibratedClassifierCV
    best_alpha: float
    validation_scores: pd.DataFrame
    test_metrics: dict[str, float]
    class_metrics: pd.DataFrame
    confusion: pd.DataFrame
    examples: pd.DataFrame

    def predict_proba(self, texts: list[str] | pd.Series) -> np.ndarray:
        matrix = self.vectorizer.transform(list(texts))
        raw = self.classifier.predict_proba(matrix)
        order = list(self.classifier.classes_)
        return raw[:, [order.index(label) for label in LABELS]]


def build_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=60_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def multiclass_brier(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    index = {label: position for position, label in enumerate(LABELS)}
    truth = np.zeros_like(probabilities)
    for row, label in enumerate(y_true):
        truth[row, index[str(label)]] = 1.0
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def top_label_ece(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    predicted_index = probabilities.argmax(axis=1)
    predictions = np.array(LABELS, dtype=object)[predicted_index]
    confidence = probabilities.max(axis=1)
    correct = predictions == np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return result


def _read_wcb_splits(directory: str | Path) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    splits = {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in ["train", "validation", "test"]
    }
    for name, split in splits.items():
        expected = {"sentences", "stance_label"}
        if not expected.issubset(split.columns):
            raise ValueError(f"WCB {name} split is missing {sorted(expected - set(split.columns))}")
    return splits


def _new_classifier(alpha: float) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        max_iter=3_000,
        tol=1e-5,
        average=True,
        random_state=RANDOM_STATE,
    )


def train_validate_tone_model(wcb_directory: str | Path) -> ToneModel:
    """Tune on the official validation split, calibrate there, and evaluate once on test."""
    splits = _read_wcb_splits(wcb_directory)
    vectorizer = build_vectorizer()
    x_train = vectorizer.fit_transform(splits["train"]["sentences"].astype(str))
    x_validation = vectorizer.transform(splits["validation"]["sentences"].astype(str))
    x_test = vectorizer.transform(splits["test"]["sentences"].astype(str))
    y_train = splits["train"]["stance_label"].astype(str)
    y_validation = splits["validation"]["stance_label"].astype(str)
    y_test = splits["test"]["stance_label"].astype(str)

    tuning_rows: list[dict[str, float]] = []
    fitted: dict[float, SGDClassifier] = {}
    for alpha in ALPHAS:
        classifier = _new_classifier(alpha).fit(x_train, y_train)
        fitted[alpha] = classifier
        validation_prediction = classifier.predict(x_validation)
        tuning_rows.append(
            {
                "alpha": alpha,
                "validation_accuracy": accuracy_score(y_validation, validation_prediction),
                "validation_macro_f1": f1_score(
                    y_validation, validation_prediction, average="macro"
                ),
                "validation_weighted_f1": f1_score(
                    y_validation, validation_prediction, average="weighted"
                ),
            }
        )
    validation_scores = pd.DataFrame(tuning_rows).sort_values("alpha")
    best_alpha = float(
        validation_scores.sort_values(
            ["validation_macro_f1", "alpha"], ascending=[False, True]
        ).iloc[0]["alpha"]
    )

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(fitted[best_alpha]), method="sigmoid"
    ).fit(x_validation, y_validation)
    raw_probability = calibrated.predict_proba(x_test)
    class_order = list(calibrated.classes_)
    probability = raw_probability[:, [class_order.index(label) for label in LABELS]]
    prediction = np.array(LABELS, dtype=object)[probability.argmax(axis=1)]

    report = classification_report(
        y_test,
        prediction,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    class_metrics = pd.DataFrame(
        [
            {
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        ]
    )
    confusion = pd.DataFrame(
        confusion_matrix(y_test, prediction, labels=LABELS),
        index=pd.Index(LABELS, name="actual"),
        columns=pd.Index(LABELS, name="predicted"),
    )
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_macro_f1": float(f1_score(y_test, prediction, average="macro")),
        "test_weighted_f1": float(f1_score(y_test, prediction, average="weighted")),
        "test_log_loss": float(log_loss(y_test, probability, labels=LABELS)),
        "test_multiclass_brier": multiclass_brier(y_test, probability),
        "test_top_label_ece_15_bins": top_label_ece(y_test, probability, bins=15),
    }
    examples = select_examples(splits["test"], y_test, prediction, probability)
    return ToneModel(
        vectorizer=vectorizer,
        classifier=calibrated,
        best_alpha=best_alpha,
        validation_scores=validation_scores,
        test_metrics=metrics,
        class_metrics=class_metrics,
        confusion=confusion,
        examples=examples,
    )


def select_examples(
    test: pd.DataFrame,
    truth: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in ["hawkish", "dovish"]:
        class_position = LABELS.index(label)
        mask = (truth.to_numpy() == label) & (prediction == label)
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            continue
        selected = candidates[np.argmax(probability[candidates, class_position])]
        record: dict[str, Any] = {
            "example_type": label,
            "sentence": str(test.iloc[selected]["sentences"]),
            "true_label": label,
            "predicted_label": str(prediction[selected]),
        }
        for position, probability_label in enumerate(LABELS):
            record[f"p_{probability_label}"] = float(probability[selected, position])
        rows.append(record)
    return pd.DataFrame(rows)


def extract_policy_clauses(text: str, max_clauses: int = 100) -> list[str]:
    """Split policy-relevant sentences at contrast markers to retain mixed scenarios."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    sentences = SENTENCE_SPLIT_RE.split(cleaned)
    clauses: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not POLICY_ANCHOR_RE.search(sentence):
            continue
        parts = CONTRAST_SPLIT_RE.split(sentence)
        usable = [part.strip() for part in parts if len(part.split()) >= 3]
        if not usable and len(sentence.split()) >= 3:
            usable = [sentence]
        for clause in usable:
            clauses.append(clause[:1500])
            if len(clauses) >= max_clauses:
                return clauses
    return clauses


def score_speeches(
    frame: pd.DataFrame,
    model: ToneModel,
    *,
    batch_size: int = 10_000,
    minimum_signal_strength: float = 0.15,
) -> pd.DataFrame:
    """Aggregate calibrated clause probabilities into interpretable speech scores."""
    clause_texts: list[str] = []
    clause_speech_positions: list[int] = []
    counts = np.zeros(len(frame), dtype=int)
    for position, text in enumerate(frame["text"].astype(str)):
        clauses = extract_policy_clauses(text)
        counts[position] = len(clauses)
        clause_texts.extend(clauses)
        clause_speech_positions.extend([position] * len(clauses))

    probability_sums = np.zeros((len(frame), len(LABELS)), dtype=float)
    positions = np.asarray(clause_speech_positions, dtype=int)
    for start in range(0, len(clause_texts), batch_size):
        stop = min(start + batch_size, len(clause_texts))
        probabilities = model.predict_proba(clause_texts[start:stop])
        np.add.at(probability_sums, positions[start:stop], probabilities)

    label_index = {label: position for position, label in enumerate(LABELS)}
    hawkish = probability_sums[:, label_index["hawkish"]]
    dovish = probability_sums[:, label_index["dovish"]]
    neutral = probability_sums[:, label_index["neutral"]]
    irrelevant = probability_sums[:, label_index["irrelevant"]]
    directional = hawkish + dovish
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.divide(hawkish, directional, out=np.full(len(frame), 0.5), where=directional > 0)
        mixedness = 1.0 - np.divide(
            np.abs(hawkish - dovish),
            directional,
            out=np.ones(len(frame)),
            where=directional > 0,
        )
        signal_strength = np.divide(
            directional,
            counts,
            out=np.zeros(len(frame)),
            where=counts > 0,
        )

    result = frame.copy()
    result["policy_clause_count"] = counts
    result["hawkish_probability_sum"] = hawkish
    result["dovish_probability_sum"] = dovish
    result["neutral_probability_sum"] = neutral
    result["irrelevant_probability_sum"] = irrelevant
    result["hawkishness_score"] = score
    result["tone_mixedness"] = mixedness
    result["tone_signal_strength"] = signal_strength
    result["tone_valid"] = (counts > 0) & (signal_strength >= minimum_signal_strength)
    result["tone_label"] = np.select(
        [
            ~result["tone_valid"],
            result["hawkishness_score"].gt(0.55),
            result["hawkishness_score"].lt(0.45),
        ],
        ["insufficient_signal", "hawkish", "dovish"],
        default="balanced",
    )
    return result
