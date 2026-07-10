"""
This module detects emotions from text using Watson NLP.
If the Watson API is unreachable, it uses a local fallback for testing.
"""

import json
import requests


def local_emotion_detector(text_to_analyze):
    """
    Local fallback emotion detector used when the API is unreachable.
    """

    text = text_to_analyze.lower()

    emotions = {
        "anger": 0,
        "disgust": 0,
        "fear": 0,
        "joy": 0,
        "sadness": 0
    }

    if any(word in text for word in ["happy", "glad", "joy", "excited"]):
        emotions["joy"] = 1
    elif any(word in text for word in ["mad", "angry", "furious"]):
        emotions["anger"] = 1
    elif any(word in text for word in ["disgusted", "disgust", "gross"]):
        emotions["disgust"] = 1
    elif any(word in text for word in ["sad", "unhappy", "depressed"]):
        emotions["sadness"] = 1
    elif any(word in text for word in ["afraid", "scared", "fear"]):
        emotions["fear"] = 1
    else:
        emotions["joy"] = 1

    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant_emotion
    }


def emotion_detector(text_to_analyze):
    """
    Detect emotions in the given text using Watson NLP.
    """
    if not text_to_analyze:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }


    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    input_json = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        response = requests.post(url, json=input_json, headers=headers, timeout=10)

        if response.status_code == 400:
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None
            }

        formatted_response = json.loads(response.text)
        emotions = formatted_response["emotionPredictions"][0]["emotion"]

        emotion_scores = {
            "anger": emotions["anger"],
            "disgust": emotions["disgust"],
            "fear": emotions["fear"],
            "joy": emotions["joy"],
            "sadness": emotions["sadness"]
        }

        dominant_emotion = max(emotion_scores, key=emotion_scores.get)

        return {
            "anger": emotions["anger"],
            "disgust": emotions["disgust"],
            "fear": emotions["fear"],
            "joy": emotions["joy"],
            "sadness": emotions["sadness"],
            "dominant_emotion": dominant_emotion
        }

    except requests.exceptions.RequestException:
        return local_emotion_detector(text_to_analyze)
