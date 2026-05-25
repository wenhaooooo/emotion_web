import os
import logging
import tempfile
from typing import Optional

import numpy as np
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)

EMOTION2VEC_LABELS = {
    0: "angry", 1: "disgusted", 2: "fearful", 3: "happy",
    4: "neutral", 5: "other", 6: "sad", 7: "surprised", 8: "unknown",
}

EMOTION2VEC_TO_TEACHER = {
    0: 3, 1: 2, 2: 3, 3: 0, 4: 1, 5: 1, 6: 2, 7: 0, 8: 1,
}

TEACHER_LABELS = {
    0: "enthusiastic", 1: "calm", 2: "negative", 3: "tense",
}

TEACHER_LABELS_CN = {
    0: "热情投入", 1: "平稳中性", 2: "消极低落", 3: "紧张焦虑",
}

_emotion2vec_cache = {}


def _get_predictor(model_id: str = "iic/emotion2vec_plus_large", hub: str = "ms", model_path: str = None):
    cache_key = (model_id, hub, model_path)
    if cache_key in _emotion2vec_cache:
        return _emotion2vec_cache[cache_key]
    predictor = Emotion2VecPredictor(model_id=model_id, hub=hub, model_path=model_path)
    _emotion2vec_cache[cache_key] = predictor
    return predictor


class Emotion2VecPredictor:
    def __init__(self, model_id="iic/emotion2vec_plus_large", hub="ms", model_path=None):
        from funasr import AutoModel
        if model_path and os.path.isdir(model_path):
            logger.info(f"Loading local emotion2vec model: {model_path}")
            self.model = AutoModel(model=model_path)
        else:
            logger.info(f"Loading emotion2vec model: {model_id}")
            self.model = AutoModel(model=model_id, hub=hub)
        logger.info("emotion2vec model loaded")

    def predict(self, wav_path: str) -> dict:
        """Predict emotion from a WAV file. Returns dict with label_9, label_4, confidence, etc."""
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            return self._empty()

        try:
            rec_result = self.model.generate(wav_path, granularity="utterance", extract_embedding=False)
            result = rec_result[0] if isinstance(rec_result, list) and rec_result else rec_result
            if not isinstance(result, dict):
                return self._empty()

            labels = result.get("labels", ["neutral"])
            scores = result.get("scores", [0.0])

            if isinstance(scores, list) and len(scores) == 9:
                scores_9 = {i: float(scores[i]) for i in range(9)}
                max_idx = max(scores_9, key=scores_9.get)
                score = scores_9[max_idx]
            else:
                score = float(scores[0]) if scores else 0.0
                max_idx = 4
                scores_9 = {i: (score if i == max_idx else 0.0) for i in range(9)}

            # Resolve label index
            if isinstance(labels, list) and len(labels) > max_idx:
                label_val = labels[max_idx]
            else:
                label_val = max_idx

            if isinstance(label_val, str):
                label_str = label_val.split("/")[-1].strip().lower()
                label_9 = next((k for k, v in EMOTION2VEC_LABELS.items() if v == label_str), max_idx)
            else:
                label_9 = int(label_val) if isinstance(label_val, (int, np.integer)) else max_idx

            label_4 = EMOTION2VEC_TO_TEACHER.get(label_9, 1)

            return {
                "label_9": label_9,
                "label_name_9": EMOTION2VEC_LABELS.get(label_9, "unknown"),
                "label_4": label_4,
                "label_name_4": TEACHER_LABELS.get(label_4, "calm"),
                "label_name_4_cn": TEACHER_LABELS_CN.get(label_4, "平稳中性"),
                "confidence": round(float(score), 4),
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._empty()

    def _empty(self):
        return {
            "label_9": 8, "label_name_9": "unknown",
            "label_4": 1, "label_name_4": "calm", "label_name_4_cn": "平稳中性",
            "confidence": 0.0,
        }


def recognize_segment(video_path: str, model_id="iic/emotion2vec_plus_large", hub="ms", model_path=None) -> dict:
    """Extract audio from video segment and run emotion2vec prediction."""
    from core.ffmpeg_utils import extract_audio

    wav_path = extract_audio(video_path)
    if wav_path is None:
        return {"error": "Failed to extract audio"}

    try:
        predictor = _get_predictor(model_id=model_id, hub=hub, model_path=model_path)
        return predictor.predict(wav_path)
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
