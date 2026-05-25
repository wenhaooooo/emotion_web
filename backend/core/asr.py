import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

_paraformer_model = None

PUNCTUATION = set('。！？，、；：,.!?;:')


def _get_paraformer_model():
    global _paraformer_model
    if _paraformer_model is not None:
        return _paraformer_model

    from funasr import AutoModel
    logger.info("Loading Paraformer model (VAD + punctuation)...")
    _paraformer_model = AutoModel(
        model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        vad_model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        punc_model='iic/punc_ct-transformer_cn-en-common-vocab471067-large',
        hub='ms',
        disable_update=True,
    )
    logger.info("Paraformer model loaded")
    return _paraformer_model


def transcribe(audio_path: str, language: str = "zh") -> List[Dict]:
    """
    Transcribe audio using FunASR Paraformer with VAD and punctuation.

    Returns a list of sentence segments: [{"start": float, "end": float, "text": str}, ...]
    Times are in seconds.
    """
    model = _get_paraformer_model()

    result = model.generate(input=audio_path, batch_size_s=300)
    if not result:
        return []

    r = result[0]
    text_punct = r.get("text", "")
    timestamps = r.get("timestamp", [])

    if not text_punct or not timestamps:
        return []

    # Map punctuated text characters to timestamps
    char_timestamps = _map_chars_to_timestamps(text_punct, timestamps)

    # Split text by sentence-ending punctuation into sentences
    sentences = _split_sentences(text_punct, char_timestamps)

    logger.info(f"Paraformer transcribed {len(sentences)} sentences from {audio_path}")
    return sentences


def _map_chars_to_timestamps(text: str, timestamps: list) -> list:
    """Map each character in punctuated text to its [start_ms, end_ms]."""
    ts_idx = 0
    char_ts = []

    for ch in text:
        if ch in PUNCTUATION:
            # Punctuation: reuse the previous character's end time
            if ts_idx > 0:
                char_ts.append(timestamps[ts_idx - 1])
            else:
                char_ts.append([0, 0])
        else:
            if ts_idx < len(timestamps):
                char_ts.append(timestamps[ts_idx])
                ts_idx += 1
            else:
                char_ts.append([0, 0])

    return char_ts


def _split_sentences(text: str, char_timestamps: list) -> list:
    """Split punctuated text by sentence-ending marks and map to timestamps."""
    # Split by sentence-ending punctuation, keeping the delimiter
    parts = re.split(r'([。！？])', text)

    # Recombine: "sentence" + "punctuation" pairs
    combined = []
    for i in range(0, len(parts) - 1, 2):
        s = parts[i]
        if i + 1 < len(parts):
            s += parts[i + 1]
        if s.strip():
            combined.append(s)
    # Trailing text without ending punctuation
    if len(parts) % 2 == 1 and parts[-1].strip():
        combined.append(parts[-1])

    # Map each sentence to its start/end timestamps
    sentences = []
    pos = 0
    for sent in combined:
        start_pos = pos
        end_pos = pos + len(sent)

        start_ms = char_timestamps[start_pos][0]
        end_ms = char_timestamps[end_pos - 1][1]

        # Remove punctuation from the stored text (clean for display)
        clean_text = re.sub(r'[。！？，、；：]', '', sent).strip()

        if clean_text:
            sentences.append({
                "start": start_ms / 1000.0,
                "end": end_ms / 1000.0,
                "text": clean_text,
            })

        pos = end_pos

    return sentences
