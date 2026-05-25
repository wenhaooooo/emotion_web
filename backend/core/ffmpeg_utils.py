import os
import subprocess
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str = None, sample_rate: int = 16000) -> Optional[str]:
    """Extract audio from video as 16kHz mono WAV."""
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg audio extraction failed: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
            return None
        if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1000:
            logger.error("Extracted audio file is empty or too small")
            return None
        return output_path
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg audio extraction timed out")
        return None


def cut_video(video_path: str, start: float, end: float, output_path: str) -> bool:
    """Cut a segment from video using ffmpeg. Tries stream copy first, falls back to re-encode."""
    duration = end - start

    # Try stream copy first (fast)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "1",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            return True

        # Fallback: re-encode
        cmd_reencode = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
            "-avoid_negative_ts", "1",
            output_path,
        ]
        result2 = subprocess.run(
            cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return result2.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0

    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg cut timed out: {output_path}")
        return False


def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode == 0:
            return float(result.stdout.decode().strip())
        return None
    except Exception:
        return None
