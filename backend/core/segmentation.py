import os
import logging
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from core.ffmpeg_utils import extract_audio, cut_video

logger = logging.getLogger(__name__)

_whisper_model_cache = {}


@dataclass
class SegmentInfo:
    index: int
    start_time: float
    end_time: float
    text: str
    output_path: Optional[str] = None


def _get_whisper_model(model_size: str = "base"):
    if model_size in _whisper_model_cache:
        return _whisper_model_cache[model_size]
    try:
        import whisper
        logger.info(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        _whisper_model_cache[model_size] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        return None


def split_video_by_sentences(
    video_path: str,
    output_dir: str,
    max_segment_duration: int = 90,
    min_segment_duration: int = 10,
    whisper_model_size: str = "base",
    whisper_language: str = "zh",
    progress_callback=None,
) -> List[SegmentInfo]:
    """
    Split video by sentence boundaries using Whisper ASR timestamps.

    Returns list of SegmentInfo with start/end times and ASR text.
    """
    os.makedirs(output_dir, exist_ok=True)

    whisper_model = _get_whisper_model(whisper_model_size)
    if whisper_model is None:
        raise RuntimeError("Failed to load Whisper model")

    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # Step 1: Extract audio
    if progress_callback:
        progress_callback("Extracting audio...")
    tmp_audio = extract_audio(video_path)
    if tmp_audio is None:
        raise RuntimeError("Failed to extract audio from video")

    # Step 2: Whisper ASR
    if progress_callback:
        progress_callback("Running speech recognition...")
    try:
        asr_result = whisper_model.transcribe(
            tmp_audio, language=whisper_language, fp16=False, word_timestamps=True,
            temperature=0.0, compression_ratio_threshold=2.4, logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)

    segments_info = asr_result.get("segments", [])
    if not segments_info:
        raise RuntimeError("No speech detected in video")

    # Step 3: Build sentence list and merge into segments
    sentences = []
    for seg in segments_info:
        start, end, text = seg["start"], seg["end"], seg["text"].strip()
        if text:
            sentences.append((start, end, text))

    if not sentences:
        raise RuntimeError("No valid sentences from ASR")

    # Merge consecutive sentences into segments (max max_segment_duration)
    merged = []
    cur_start, cur_end, cur_texts = sentences[0][0], sentences[0][1], [sentences[0][2]]

    for i in range(1, len(sentences)):
        s_start, s_end, s_text = sentences[i]
        if s_end - cur_start > max_segment_duration:
            merged.append((cur_start, cur_end, cur_texts))
            cur_start, cur_end, cur_texts = s_start, s_end, [s_text]
        else:
            cur_end = s_end
            cur_texts.append(s_text)
    merged.append((cur_start, cur_end, cur_texts))

    # Step 4: Cut video segments
    if progress_callback:
        progress_callback("Cutting video segments...")
    results = []
    for idx, (start_time, end_time, texts) in enumerate(merged):
        seg_duration = end_time - start_time
        if seg_duration < min_segment_duration:
            continue

        out_filename = f"{video_basename}_seg{idx:04d}.mp4"
        out_path = os.path.join(output_dir, out_filename)
        combined_text = " ".join(texts)

        if cut_video(video_path, start_time, end_time, out_path):
            results.append(SegmentInfo(
                index=idx,
                start_time=start_time,
                end_time=end_time,
                text=combined_text,
                output_path=out_path,
            ))

    logger.info(f"Split into {len(results)} segments")
    return results


def split_video_by_utterances(
    video_path: str,
    output_dir: str,
    min_segment_duration: float = 2.0,
    progress_callback=None,
) -> List[SegmentInfo]:
    """
    Split video by individual ASR sentence boundaries using FunASR Paraformer.

    Each sentence (split by punctuation) becomes its own clip — no merging.
    This preserves fine-grained emotion changes within seconds.
    """
    from core.asr import transcribe

    os.makedirs(output_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # Step 1: Extract audio
    if progress_callback:
        progress_callback("Extracting audio...")
    tmp_audio = extract_audio(video_path)
    if tmp_audio is None:
        raise RuntimeError("Failed to extract audio from video")

    # Step 2: Paraformer ASR with punctuation
    if progress_callback:
        progress_callback("Running speech recognition (Paraformer)...")
    try:
        sentences = transcribe(tmp_audio)
    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)

    if not sentences:
        raise RuntimeError("No speech detected in video")

    # Step 3: Cut each sentence as its own segment
    if progress_callback:
        progress_callback("Cutting video segments...")
    results = []
    for idx, sent in enumerate(sentences):
        start_time = sent["start"]
        end_time = sent["end"]
        text = sent["text"]

        if not text:
            continue
        if (end_time - start_time) < min_segment_duration:
            continue

        out_filename = f"{video_basename}_seg{idx:04d}.mp4"
        out_path = os.path.join(output_dir, out_filename)

        if cut_video(video_path, start_time, end_time, out_path):
            results.append(SegmentInfo(
                index=idx,
                start_time=start_time,
                end_time=end_time,
                text=text,
                output_path=out_path,
            ))

    logger.info(f"Split into {len(results)} utterance segments")
    return results
