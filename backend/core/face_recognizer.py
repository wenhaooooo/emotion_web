"""
面部表情识别模块
使用 HuggingFace Transformers 加载预训练模型，对人脸图像进行 7 类情感分类。
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# 7 类情感标签（与 FER 标准一致）
FACE_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# emotion2vec 9 类 → 统一 7 类映射
EMOTION2VEC_TO_7CLASS = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprise",
    "other": "neutral",
    "unknown": "neutral",
}

# 7 类情感中文名
EMOTION_NAMES_CN = {
    "angry": "愤怒",
    "disgust": "厌恶",
    "fear": "恐惧",
    "happy": "快乐",
    "neutral": "中性",
    "sad": "悲伤",
    "surprise": "惊讶",
}

# 7 类情感图标
EMOTION_ICONS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

# 7 类情感色值
EMOTION_COLORS = {
    "angry": "#FF3B30",
    "disgust": "#8B4513",
    "fear": "#AF52DE",
    "happy": "#34C759",
    "neutral": "#8E8E93",
    "sad": "#007AFF",
    "surprise": "#FFCC00",
}

# 模块级单例
_face_recognizer_instance: Optional["FaceRecognizer"] = None


def get_face_recognizer() -> "FaceRecognizer":
    """获取面部识别器单例"""
    global _face_recognizer_instance
    if _face_recognizer_instance is None:
        _face_recognizer_instance = FaceRecognizer()
    return _face_recognizer_instance


def init_face_recognizer():
    """初始化并加载面部识别器（在应用启动时调用）"""
    recognizer = get_face_recognizer()
    recognizer.load()
    return recognizer


class FaceRecognizer:
    """面部表情识别器"""

    def __init__(self):
        self.classifier = None
        self.face_cascade = None
        self._loaded = False

    def load(self):
        """加载模型（应用启动时调用）"""
        if self._loaded:
            return

        import sys
        from transformers import pipeline

        logger.info("正在加载面部表情识别模型...")

        # 静默加载：transformers 会输出 Loading weights 等进度条
        _devnull = open(os.devnull, "w")
        _old_stdout, _old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _devnull
        try:
            self.classifier = pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection",
            )
        finally:
            sys.stdout, sys.stderr = _old_stdout, _old_stderr
            _devnull.close()

        logger.info("面部表情识别模型加载完成")

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            logger.error("OpenCV Haar Cascade 加载失败")
            raise RuntimeError("Failed to load Haar Cascade")

        self._loaded = True
        logger.info("面部识别模块初始化完成")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        检测画面中的人脸区域。

        Args:
            frame: BGR 格式的 numpy 数组

        Returns:
            人脸矩形列表 [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48)
        )
        return list(faces) if len(faces) > 0 else []

    def predict(self, frame: np.ndarray) -> dict:
        """
        对画面进行面部表情识别。

        Args:
            frame: BGR 格式的 numpy 数组

        Returns:
            {
                "face_detected": True/False,
                "emotion": "happy",
                "confidence": 0.873,
                "probabilities": { ... },
                "face_box": [x, y, w, h]
            }
        """
        if not self._loaded:
            return {"face_detected": False}

        faces = self.detect_faces(frame)
        if len(faces) == 0:
            return {"face_detected": False}

        # 取最大的人脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # 裁剪人脸区域，转为 PIL Image (RGB)
        face_roi = frame[y : y + h, x : x + w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)

        try:
            # HuggingFace 推理
            results = self.classifier(pil_image)

            # 构建概率字典
            probabilities = {r["label"]: round(r["score"], 4) for r in results}
            top = results[0]

            return {
                "face_detected": True,
                "emotion": top["label"],
                "confidence": round(top["score"], 4),
                "probabilities": probabilities,
                "face_box": [int(x), int(y), int(w), int(h)],
            }
        except Exception as e:
            logger.error(f"面部表情识别失败: {e}")
            return {"face_detected": False}


def map_emotion2vec_to_7class(emotion2vec_result: dict) -> dict:
    """
    将 emotion2vec 的 9 类结果映射为统一 7 类标签。

    Args:
        emotion2vec_result: emotion2vec 原始输出
            {"label_name_9": "happy", "confidence": 0.9, ...}

    Returns:
        映射后的 7 类结果
    """
    label_9_name = emotion2vec_result.get("label_name_9", "unknown")
    label_7 = EMOTION2VEC_TO_7CLASS.get(label_9_name, "neutral")
    confidence = emotion2vec_result.get("confidence", 0.0)

    # 构建 7 类概率（emotion2vec 不直接输出 7 类概率，只输出最高类）
    probabilities = {e: 0.0 for e in FACE_EMOTIONS}
    probabilities[label_7] = confidence
    # 剩余概率均分给其他类
    remaining = 1.0 - confidence
    other_count = len(FACE_EMOTIONS) - 1
    if other_count > 0:
        for e in FACE_EMOTIONS:
            if e != label_7:
                probabilities[e] = round(remaining / other_count, 4)

    return {
        "emotion": label_7,
        "confidence": confidence,
        "probabilities": probabilities,
    }
