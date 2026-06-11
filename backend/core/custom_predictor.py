"""
自训练模型的教师情感预测器 — 完全解耦版本。

使用 output/ablation_results_10-fold 中最好的模型 (full_model, fold_9)。
三模态输入: text (768-d RoBERTa CLS), audio (1024-d emotion2vec), vision (28-d MediaPipe)
输出: 4 类教师情感 (热情投入 / 平稳中性 / 消极低落 / 紧张焦虑)

本模块自包含所有特征提取逻辑，不依赖主项目 src/ 下的代码。
"""

import os
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ============================================================
# 标签定义
# ============================================================

CUSTOM_LABELS = {
    0: "enthusiastic", 1: "calm", 2: "negative", 3: "anxious",
}

CUSTOM_LABELS_CN = {
    0: "热情投入", 1: "平稳中性", 2: "消极低落", 3: "紧张焦虑",
}

# 用于映射到 emotion2vec 9 类标签的虚拟映射（保持接口兼容）
_LABEL_TO_EMOTION2VEC = {
    0: 3,   # enthusiastic -> happy
    1: 4,   # calm -> neutral
    2: 6,   # negative -> sad
    3: 0,   # anxious -> angry
}

EMOTION2VEC_LABELS = {
    0: "angry", 1: "disgusted", 2: "fearful", 3: "happy",
    4: "neutral", 5: "other", 6: "sad", 7: "surprised", 8: "unknown",
}

# 模型单例缓存
_custom_cache = {}


def _get_custom_predictor(model_path=None, device='cpu'):
    """获取或创建 CustomEmotionPredictor 单例。"""
    cache_key = (model_path, device)
    if cache_key in _custom_cache:
        return _custom_cache[cache_key]
    predictor = CustomEmotionPredictor(model_path=model_path, device=device)
    _custom_cache[cache_key] = predictor
    return predictor


class CustomEmotionPredictor:
    """
    自训练多模态教师情感识别模型的预测器。

    流程:
    1. 从视频段提取音频 → emotion2vec 1024-d 音频特征
    2. 从视频段提取人脸 → MediaPipe 28-d 视觉特征
    3. ASR 文本 → RoBERTa 768-d 文本特征
    4. 三模态特征送入 PromptModel 推理 → 4 类教师情感
    """

    def __init__(self, model_path=None, device='cpu'):
        """
        初始化预测器。模型和特征提取器均为延迟加载。

        参数:
            model_path: best_model.pt 的路径。为 None 时使用默认路径。
            device: 'cpu' 或 'cuda'
        """
        self.device = torch.device(device)
        self._model = None
        self._hyp_params = None
        self._model_path = model_path
        self._bert_tokenizer = None
        self._bert_model = None
        self._emotion2vec_model = None
        self._face_mesh_model = None
        self._haar_cascade = None

        # BERT 模型路径 — 与主项目 extract_features.py 使用相同模型
        self._bert_model_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'models', 'roberta-chinese')
        # 如果相对路径不存在，尝试绝对路径
        if not os.path.isdir(self._bert_model_path):
            self._bert_model_path = os.path.abspath(self._bert_model_path)

    @property
    def is_loaded(self):
        return self._model is not None

    # ============================================================
    # 模型加载
    # ============================================================

    def _ensure_model(self):
        """延迟加载模型。"""
        if self._model is not None:
            return

        model_path = self._model_path
        if model_path is None:
            # 默认使用 emotion_web/backend/models/ 下的模型
            model_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'full_model_9th_fold.pt')

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        logger.info(f"加载自训练模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        from types import SimpleNamespace
        from core.custom_model import PromptModel

        # checkpoint 可能是两种格式:
        # 1. 包装格式: {'hyp_params': ..., 'model_state_dict': ...}
        # 2. 原始 state_dict (CV 实验保存的)
        if 'hyp_params' in checkpoint and 'model_state_dict' in checkpoint:
            hyp_params = SimpleNamespace(**checkpoint['hyp_params'])
            state_dict = checkpoint['model_state_dict']
        else:
            # 原始 state_dict — 使用硬编码的 full_model 配置
            hyp_params = SimpleNamespace(
                orig_d_l=768, orig_d_a=1024, orig_d_v=28,
                proj_dim=50, num_heads=5, layers=3,
                attn_dropout=0.0, attn_dropout_a=0.0, attn_dropout_v=0.0,
                relu_dropout=0.0, res_dropout=0.0, out_dropout=0.0, embed_dropout=0.0,
                attn_mask=False,
                prompt_length=16, prompt_dim=50,
                seq_len=(30, 30, 30),
                output_dim=4,
                use_generative_prompt=True, use_modality_signal=True,
                use_missing_type_prompt=True,
                use_temporal=True, max_seq_len=30,
                use_context=True,
                use_cross_domain=False,
            )
            state_dict = checkpoint

        model = PromptModel(hyp_params)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        self._model = model
        self._hyp_params = hyp_params
        self._max_seq_len = 30  # 训练时使用的最大序列长度
        logger.info("自训练模型加载完成 (full_model, fold_9)")

    # ============================================================
    # 文本特征提取 (RoBERTa CLS, 768-d)
    # ============================================================

    def _ensure_bert(self):
        """延迟加载 RoBERTa 模型。"""
        if self._bert_tokenizer is not None:
            return

        bert_path = self._bert_model_path
        if not os.path.isdir(bert_path):
            logger.warning(f"RoBERTa 模型目录不存在: {bert_path}，文本特征将使用零向量")
            self._bert_tokenizer = "not_available"
            return

        try:
            from transformers import RobertaTokenizer, RobertaModel
            logger.info(f"加载 RoBERTa 模型: {bert_path}")
            self._bert_tokenizer = RobertaTokenizer.from_pretrained(bert_path)
            self._bert_model = RobertaModel.from_pretrained(bert_path)
            self._bert_model.eval()
            self._bert_model.to(self.device)
            logger.info("RoBERTa 模型加载完成")
        except Exception as e:
            logger.warning(f"RoBERTa 模型加载失败: {e}，文本特征将使用零向量")
            self._bert_tokenizer = "not_available"

    def _extract_text_features(self, text):
        """
        从文本提取 RoBERTa CLS 特征 (768-d)。

        参数:
            text: ASR 识别出的文本字符串

        返回:
            np.ndarray, shape (768,), dtype float32
        """
        self._ensure_bert()

        if not text or not text.strip() or self._bert_tokenizer == "not_available":
            return np.zeros(768, dtype=np.float32)

        try:
            inputs = self._bert_tokenizer(
                text, max_length=50, padding='max_length',
                truncation=True, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self._bert_model(input_ids=input_ids, attention_mask=attention_mask)
                cls_features = outputs.last_hidden_state[:, 0, :].squeeze(0)

            return cls_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.warning(f"文本特征提取失败: {e}")
            return np.zeros(768, dtype=np.float32)

    # ============================================================
    # 音频特征提取 (emotion2vec, 1024-d)
    # ============================================================

    def _ensure_emotion2vec(self):
        """延迟加载 emotion2vec 模型。"""
        if self._emotion2vec_model is not None:
            return

        try:
            import sys
            import warnings
            _devnull = open(os.devnull, "w")
            _old_stdout, _old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = _devnull
            warnings.filterwarnings("ignore")
            try:
                from funasr import AutoModel
                self._emotion2vec_model = AutoModel(
                    model="iic/emotion2vec_plus_large", hub="ms",
                    disable_update=True, trust_remote_code=True)
            finally:
                sys.stdout, sys.stderr = _old_stdout, _old_stderr
                _devnull.close()
                warnings.resetwarnings()
            logger.info("emotion2vec 模型加载完成")
        except Exception as e:
            logger.warning(f"emotion2vec 模型加载失败: {e}，音频特征将使用零向量")
            self._emotion2vec_model = "not_available"

    def _extract_audio_features(self, wav_path):
        """
        从 WAV 文件提取 emotion2vec 表征 (1024-d)。

        参数:
            wav_path: 16kHz 单声道 WAV 文件路径

        返回:
            np.ndarray, shape (1024,), dtype float32
        """
        self._ensure_emotion2vec()

        if self._emotion2vec_model == "not_available" or not os.path.isfile(wav_path):
            return np.zeros(1024, dtype=np.float32)

        try:
            rec_result = self._emotion2vec_model.generate(
                wav_path, output_dir=None, granularity="utterance", extract_embedding=True)

            if isinstance(rec_result, list) and len(rec_result) > 0:
                rec = rec_result[0]
                if isinstance(rec, dict) and 'feats' in rec:
                    feats = rec['feats']
                    if isinstance(feats, np.ndarray):
                        return feats.flatten().astype(np.float32)
                    return np.array(feats, dtype=np.float32).flatten()

            if isinstance(rec_result, dict) and 'feats' in rec_result:
                feats = rec_result['feats']
                if isinstance(feats, np.ndarray):
                    return feats.flatten().astype(np.float32)
                return np.array(feats, dtype=np.float32).flatten()

            logger.warning(f"emotion2vec 返回格式异常: {type(rec_result)}")
            return np.zeros(1024, dtype=np.float32)
        except Exception as e:
            logger.warning(f"emotion2vec 特征提取失败: {e}")
            return np.zeros(1024, dtype=np.float32)

    # ============================================================
    # 视觉特征提取 (MediaPipe Face Mesh, 28-d)
    # ============================================================

    def _ensure_face_mesh(self):
        """延迟加载 MediaPipe Face Mesh。"""
        if self._face_mesh_model is not None:
            return

        try:
            import mediapipe as mp
            self._face_mesh_model = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.3, min_tracking_confidence=0.3)
            logger.info("MediaPipe Face Mesh 加载完成")
        except Exception as e:
            logger.warning(f"MediaPipe 加载失败: {e}，视觉特征将使用零向量")
            self._face_mesh_model = "not_available"

    def _ensure_haar_cascade(self):
        """加载 Haar 级联分类器。"""
        if self._haar_cascade is not None:
            return
        try:
            import cv2
            default_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.isfile(default_path):
                self._haar_cascade = cv2.CascadeClassifier(default_path)
            else:
                self._haar_cascade = "not_available"
        except Exception:
            self._haar_cascade = "not_available"

    def _detect_face_region(self, frame):
        """Haar 级联人脸检测并裁剪。"""
        import cv2
        self._ensure_haar_cascade()
        if self._haar_cascade == "not_available":
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            margin_x, margin_y = int(fw * 0.3), int(fh * 0.3)
            x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
            x2 = min(frame.shape[1], x + fw + margin_x)
            y2 = min(frame.shape[0], y + fh + margin_y)
            return frame[y1:y2, x1:x2]
        return frame

    @staticmethod
    def _compute_au_from_landmarks(landmarks, w, h):
        """从 MediaPipe 关键点推算 AU-like 特征 (9-d)。"""
        lm = np.array([[p.x * w, p.y * h, p.z] for p in landmarks])

        def dist(a, b):
            return np.linalg.norm(lm[a] - lm[b])

        face_height = dist(10, 152)
        if face_height < 1e-6:
            face_height = 1.0

        aus = []
        # AU4 - 皱眉
        brow_inner_dist = (dist(66, 133) + dist(296, 362)) / 2.0
        eye_inner_dist = (dist(133, 33) + dist(362, 263)) / 2.0
        aus.append((brow_inner_dist / (eye_inner_dist + 1e-6) - 1.0) * 10.0)
        # AU6 - 脸颊收紧
        cheek_tight = (dist(61, 33) + dist(291, 263)) / 2.0
        aus.append(cheek_tight / face_height * 5.0)
        # AU9 - 鼻皱
        nose_width = dist(48, 278)
        nose_bridge = dist(6, 168)
        aus.append(nose_width / (nose_bridge + 1e-6) * 5.0)
        # AU12 - 嘴角上扬
        mouth_center_y = (lm[13][1] + lm[14][1]) / 2.0
        mouth_corner_y = (lm[61][1] + lm[291][1]) / 2.0
        mouth_width = dist(61, 291)
        smile = (mouth_center_y - mouth_corner_y) / (mouth_width + 1e-6) * 10.0
        aus.append(smile)
        # AU15 - 嘴角下拉
        aus.append(max(0, -smile))
        # AU17 - 下巴抬起
        chin_raise = dist(17, 152) / face_height * 10.0
        aus.append(chin_raise)
        # AU24 - 嘴唇收紧
        lip_upper = dist(13, 0)
        lip_lower = dist(14, 17)
        lip_tight = (lip_upper + lip_lower) / (face_height + 1e-6) * 5.0
        aus.append(lip_tight)
        # AU25 - 嘴唇张开
        lip_open = dist(13, 14) / face_height * 10.0
        aus.append(lip_open)
        # AU45 - 眨眼
        left_eye_open = dist(159, 145)
        right_eye_open = dist(386, 374)
        eye_open = (left_eye_open + right_eye_open) / 2.0
        eye_width = (dist(33, 133) + dist(263, 362)) / 2.0
        aus.append(eye_open / (eye_width + 1e-6) * 5.0)

        return np.array(aus, dtype=np.float32)

    @staticmethod
    def _compute_face_geometry(landmarks, w, h):
        """从 MediaPipe 关键点头部姿态/眼睛/嘴巴几何特征 (19-d)。"""
        lm = np.array([[p.x * w, p.y * h, p.z] for p in landmarks])

        def dist(a, b):
            return np.linalg.norm(lm[a] - lm[b])

        face_height = dist(10, 152)
        if face_height < 1e-6:
            face_height = 1.0

        feats = []
        # Head Pose (3-d)
        nose_tip = lm[1]
        left_face, right_face = lm[234], lm[454]
        face_center_x = (left_face[0] + right_face[0]) / 2.0
        face_width = abs(right_face[0] - left_face[0]) + 1e-6
        feats.append((nose_tip[0] - face_center_x) / face_width * 2.0)
        forehead, chin = lm[10], lm[152]
        feats.append((nose_tip[1] - forehead[1]) / (chin[1] - forehead[1] + 1e-6) * 2.0 - 1.0)
        left_eye_center = (lm[33] + lm[133]) / 2.0
        right_eye_center = (lm[263] + lm[362]) / 2.0
        feats.append(np.arctan2(right_eye_center[1] - left_eye_center[1],
                                right_eye_center[0] - left_eye_center[0]))
        # Gaze (2-d)
        left_iris = lm[468] if len(lm) > 468 else lm[0]
        left_eye_width = dist(133, 33) + 1e-6
        feats.append((left_iris[0] - lm[133][0]) / left_eye_width * 2.0 - 1.0)
        right_iris = lm[473] if len(lm) > 473 else lm[0]
        right_eye_width = dist(362, 263) + 1e-6
        feats.append((right_iris[0] - lm[362][0]) / right_eye_width * 2.0 - 1.0)
        # Eye (4-d)
        feats.append(dist(159, 145) / face_height * 10.0)
        feats.append(dist(386, 374) / face_height * 10.0)
        feats.append(dist(33, 133) / face_height * 10.0)
        feats.append(dist(263, 362) / face_height * 10.0)
        # Eyebrow (4-d)
        feats.append(dist(70, 159) / face_height * 10.0)
        feats.append(dist(300, 386) / face_height * 10.0)
        feats.append(dist(107, 70) / face_height * 10.0)
        feats.append(dist(336, 300) / face_height * 10.0)
        # Mouth (6-d)
        feats.append(dist(13, 14) / face_height * 10.0)
        feats.append(dist(61, 291) / face_height * 10.0)
        feats.append(dist(0, 13) / face_height * 10.0)
        feats.append(dist(17, 14) / face_height * 10.0)
        mouth_center_y = (lm[13][1] + lm[14][1]) / 2.0
        mouth_corner_y = (lm[61][1] + lm[291][1]) / 2.0
        feats.append((mouth_center_y - mouth_corner_y) / face_height * 10.0)
        feats.append(dist(78, 308) / face_height * 10.0)

        return np.array(feats, dtype=np.float32)

    def _extract_visual_features(self, video_path, max_frames=50):
        """
        从视频提取 MediaPipe 视觉特征 (28-d)。

        对 max_frames 帧均匀采样，提取面部关键点特征后取平均。

        参数:
            video_path: 视频文件路径
            max_frames: 最大采样帧数

        返回:
            np.ndarray, shape (28,), dtype float32
        """
        import cv2
        self._ensure_face_mesh()

        if self._face_mesh_model == "not_available":
            return np.zeros(28, dtype=np.float32)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros(28, dtype=np.float32)

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return np.zeros(28, dtype=np.float32)

            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            all_features = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                face_crop = self._detect_face_region(frame)
                crop_h, crop_w = face_crop.shape[:2]

                rgb_frame = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                results = self._face_mesh_model.process(rgb_frame)

                if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                    landmarks = results.multi_face_landmarks[0].landmark
                    au_feats = self._compute_au_from_landmarks(landmarks, crop_w, crop_h)
                    geo_feats = self._compute_face_geometry(landmarks, crop_w, crop_h)
                    frame_feat = np.concatenate([geo_feats, au_feats]).astype(np.float32)
                else:
                    frame_feat = np.zeros(28, dtype=np.float32)

                all_features.append(frame_feat)

            if len(all_features) == 0:
                return np.zeros(28, dtype=np.float32)

            return np.mean(all_features, axis=0).astype(np.float32)
        except Exception as e:
            logger.error(f"视觉特征提取失败: {e}")
            return np.zeros(28, dtype=np.float32)
        finally:
            cap.release()

    # ============================================================
    # 模型推理
    # ============================================================

    def _run_inference(self, text_feat, audio_feat, vision_feat, missing_mod=6):
        """
        运行 PromptModel 推理。

        参数:
            text_feat: np.ndarray, shape (768,) 或 (seq_len, 768)
            audio_feat: np.ndarray, shape (1024,) 或 (seq_len, 1024)
            vision_feat: np.ndarray, shape (28,) 或 (seq_len, 28)
            missing_mod: 缺失模态模式 (6=完整)

        返回:
            dict: label_4, label_name_4, label_name_4_cn, confidence,
                  label_9, label_name_9, probabilities
        """
        self._ensure_model()
        max_seq_len = self._max_seq_len  # 30 (训练时的序列长度)

        # 确保 2D: (seq_len, dim)
        if text_feat.ndim == 1:
            text_feat = text_feat[np.newaxis, :]
        if audio_feat.ndim == 1:
            audio_feat = audio_feat[np.newaxis, :]
        if vision_feat.ndim == 1:
            vision_feat = vision_feat[np.newaxis, :]

        # 记录实际 seq_len，然后 pad 到 max_seq_len
        actual_seq_len = text_feat.shape[0]

        def pad_to_max(feat, max_len):
            """将特征 pad 到 max_len 维度。"""
            cur_len = feat.shape[0]
            if cur_len >= max_len:
                return feat[:max_len]
            pad = np.zeros((max_len - cur_len, feat.shape[1]), dtype=feat.dtype)
            return np.concatenate([feat, pad], axis=0)

        text_feat = pad_to_max(text_feat, max_seq_len)
        audio_feat = pad_to_max(audio_feat, max_seq_len)
        vision_feat = pad_to_max(vision_feat, max_seq_len)

        # 添加 batch 维度: (seq_len, dim) -> (1, seq_len, dim)
        text_feat = text_feat[np.newaxis, :]
        audio_feat = audio_feat[np.newaxis, :]
        vision_feat = vision_feat[np.newaxis, :]

        text_t = torch.FloatTensor(text_feat).to(self.device)
        audio_t = torch.FloatTensor(audio_feat).to(self.device)
        vision_t = torch.FloatTensor(vision_feat).to(self.device)
        missing_mod_t = torch.LongTensor([missing_mod]).to(self.device)
        seq_lengths = torch.LongTensor([actual_seq_len]).to(self.device)

        with torch.no_grad():
            output, _, _ = self._model(
                text_t, audio_t, vision_t,
                missing_mod=missing_mod_t, seq_lengths=seq_lengths)

        probs = torch.softmax(output, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        # 映射到 emotion2vec 9 类（接口兼容）
        e2v_label = _LABEL_TO_EMOTION2VEC.get(pred, 4)

        return {
            "label_4": pred,
            "label_name_4": CUSTOM_LABELS.get(pred, "calm"),
            "label_name_4_cn": CUSTOM_LABELS_CN.get(pred, "平稳中性"),
            "confidence": round(conf, 4),
            "label_9": e2v_label,
            "label_name_9": EMOTION2VEC_LABELS.get(e2v_label, "unknown"),
            "probabilities": {CUSTOM_LABELS[i]: round(float(probs[i]), 4) for i in range(4)},
        }

    # ============================================================
    # 主预测接口
    # ============================================================

    def predict(self, video_path, wav_path=None, text=None):
        """
        从视频段预测教师情感。

        参数:
            video_path: 视频段文件路径
            wav_path: 已提取的 WAV 文件路径（可选，避免重复提取）
            text: ASR 识别文本（可选，如果不提供则返回零文本特征）

        返回:
            dict: 与 Emotion2VecPredictor.predict() 兼容的格式
                  包含 label_9, label_name_9, label_4, label_name_4, label_name_4_cn, confidence
        """
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
            return self._empty()

        try:
            # 1. 提取文本特征 (768-d)
            text_feat = self._extract_text_features(text or "")

            # 2. 提取音频特征 (1024-d)
            audio_feat = self._extract_audio_features(wav_path or video_path)

            # 3. 提取视觉特征 (28-d)
            vision_feat = self._extract_visual_features(video_path)

            # 4. 判断缺失模态模式
            has_text = text and text.strip()
            has_audio = np.any(audio_feat != 0)
            has_vision = np.any(vision_feat != 0)

            if has_text and has_audio and has_vision:
                missing_mod = 6  # 完整
            elif has_audio and has_vision:
                missing_mod = 0  # 缺文本
            elif has_text and has_vision:
                missing_mod = 1  # 缺音频
            elif has_text and has_audio:
                missing_mod = 2  # 缺视觉
            elif has_vision:
                missing_mod = 3  # 缺文本+音频
            elif has_audio:
                missing_mod = 4  # 缺文本+视觉
            elif has_text:
                missing_mod = 5  # 缺音频+视觉
            else:
                missing_mod = 6  # 全缺时用完整模式（模型自己处理零输入）

            # 5. 推理
            return self._run_inference(text_feat, audio_feat, vision_feat, missing_mod)

        except Exception as e:
            logger.error(f"自训练模型预测失败: {e}", exc_info=True)
            return self._empty()

    def _empty(self):
        return {
            "label_9": 8, "label_name_9": "unknown",
            "label_4": 1, "label_name_4": "calm", "label_name_4_cn": "平稳中性",
            "confidence": 0.0,
        }


# ============================================================
# 模块级便捷函数（与 recognize_segment 接口一致）
# ============================================================

def recognize_segment_custom(video_path, wav_path=None, text=None) -> dict:
    """
    使用自训练模型对视频段进行情感识别。

    接口与 core.recognizer.recognize_segment 完全一致，可直接替换。

    参数:
        video_path: 视频段文件路径
        wav_path: 已提取的 WAV 文件路径（可选）
        text: ASR 识别文本（可选）

    返回:
        dict: 包含 label_9, label_name_9, label_4, label_name_4, label_name_4_cn, confidence
    """
    from core.ffmpeg_utils import extract_audio

    predictor = _get_custom_predictor()

    # 如果没有提供 wav_path，从视频提取
    if wav_path is None:
        wav_path = extract_audio(video_path)
        should_cleanup = wav_path is not None
    else:
        should_cleanup = False

    try:
        return predictor.predict(video_path, wav_path=wav_path, text=text)
    finally:
        if should_cleanup and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
