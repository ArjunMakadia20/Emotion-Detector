"""
Inference for the custom scikit-learn emotion classifier.

Loads the model saved by train_classifier.py and exposes a single
predict_emotion() function. The return shape deliberately mirrors the Watson
NLP output in EmotionDetection.emotion_detection (per-class scores plus a
dominant_emotion key) so the two approaches can be compared side by side.
"""

import os

import joblib

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_classifier.pkl")
LABELS = ["anger", "disgust", "fear", "joy", "sadness"]

_model = None


def _load_model():
    """Lazily load and cache the trained pipeline."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_classifier.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_emotion(text):
    """
    Predict the emotion of a piece of text.

    Returns a dict with a probability for each of the five emotions and a
    dominant_emotion key holding the highest-scoring label.
    """
    model = _load_model()
    probabilities = model.predict_proba([text])[0]
    scores = {
        label: round(float(prob), 4)
        for label, prob in zip(model.classes_, probabilities)
    }

    result = {emotion: scores.get(emotion, 0.0) for emotion in LABELS}
    result["dominant_emotion"] = max(scores, key=scores.get)
    return result
