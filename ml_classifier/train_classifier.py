"""
Train a custom scikit-learn emotion classifier.

This is a lightweight, self-contained complement to the Watson NLP integration
in the EmotionDetection package -- a from-scratch model trained on a small,
self-authored dataset so the two approaches can be compared side by side.

It loads the labeled dataset, trains TF-IDF + LogisticRegression (and
TF-IDF + MultinomialNB for comparison), evaluates both on a held-out test set,
and saves the better-performing pipeline to emotion_classifier.pkl. The default
is LogisticRegression unless Naive Bayes is clearly better.
"""

import csv
import os
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "emotion_dataset.csv")
MODEL_PATH = os.path.join(HERE, "emotion_classifier.pkl")

RANDOM_STATE = 42
LABELS = ["anger", "disgust", "fear", "joy", "sadness"]

# MultinomialNB is only preferred over LogisticRegression if it beats it by
# more than this margin in test accuracy; otherwise we default to LogReg.
SELECTION_MARGIN = 0.01


def load_dataset(path):
    """Read the labeled CSV into parallel lists of texts and labels."""
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def build_pipeline(classifier):
    """TF-IDF (unigrams + bigrams) feeding the given classifier."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", classifier),
    ])


def main():
    texts, labels = load_dataset(DATASET_PATH)
    print(f"Loaded {len(texts)} labeled examples")
    for label in LABELS:
        print(f"  {label:<8} {Counter(labels)[label]}")
    print()

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(x_train)}  Test: {len(x_test)}")
    print()

    candidates = {
        "LogisticRegression": build_pipeline(LogisticRegression(max_iter=1000)),
        "MultinomialNB": build_pipeline(MultinomialNB()),
    }

    fitted, scores = {}, {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(x_test))
        fitted[name], scores[name] = pipeline, acc
        print(f"{name:<20} test accuracy: {acc:.3f}")
    print()

    best_name = "LogisticRegression"
    if scores["MultinomialNB"] > scores["LogisticRegression"] + SELECTION_MARGIN:
        best_name = "MultinomialNB"
    best_model = fitted[best_name]
    print(f"Selected model: {best_name}")
    print()

    y_pred = best_model.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print("Order:", LABELS)
    print(confusion_matrix(y_test, y_pred, labels=LABELS))
    print()

    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
