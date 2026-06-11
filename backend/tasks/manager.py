import uuid
import logging
import threading
import json
from pathlib import Path
from typing import Optional
from queue import Queue

from api.schemas import JobStatus, JobResult, SegmentResult

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self._jobs: dict[str, JobStatus] = {}
        self._results: dict[str, JobResult] = {}
        self._video_paths: dict[str, str] = {}
        self._queues: dict[str, list[Queue]] = {}
        self._lock = threading.Lock()

    def create_job(self, video_path: str, video_name: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self._jobs[job_id] = JobStatus(job_id=job_id, status="pending", progress=0.0, message="Queued")
            self._video_paths[job_id] = video_path
            self._queues[job_id] = []

        return job_id

    def get_status(self, job_id: str) -> Optional[JobStatus]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_result(self, job_id: str) -> Optional[JobResult]:
        with self._lock:
            return self._results.get(job_id)

    def get_video_path(self, job_id: str) -> Optional[str]:
        with self._lock:
            return self._video_paths.get(job_id)

    def update_progress(self, job_id: str, progress: float, message: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress = progress
                self._jobs[job_id].message = message
                self._jobs[job_id].status = "processing"
                self._notify_sse(job_id)

    def set_done(self, job_id: str, result: JobResult):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "done"
                self._jobs[job_id].progress = 1.0
                self._jobs[job_id].message = "Done"
                self._results[job_id] = result
                self._notify_sse(job_id)

    def set_failed(self, job_id: str, error: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "failed"
                self._jobs[job_id].message = error
                self._notify_sse(job_id)

    def subscribe_sse(self, job_id: str) -> Queue:
        q = Queue()
        with self._lock:
            if job_id in self._queues:
                self._queues[job_id].append(q)
        return q

    def _notify_sse(self, job_id: str):
        status = self._jobs.get(job_id)
        if not status or job_id not in self._queues:
            return
        data = json.dumps(status.model_dump())
        dead = []
        for q in self._queues[job_id]:
            try:
                q.put_nowait(data)
            except Exception:
                dead.append(q)
        for q in dead:
            self._queues[job_id].remove(q)

    def cleanup_sse(self, job_id: str, q: Queue):
        with self._lock:
            if job_id in self._queues and q in self._queues[job_id]:
                self._queues[job_id].remove(q)


def run_processing_job(job_manager: JobManager, job_id: str, video_path: str, video_name: str, jobs_dir: Path):
    """Background thread: split video → run emotion recognition → save results."""
    from core.segmentation import split_video_by_utterances
    from core.ffmpeg_utils import get_video_duration
    import os

    # 优先使用自训练模型，回退到 emotion2vec
    use_custom = os.environ.get("EMOTION_MODEL", "custom") == "custom"
    if use_custom:
        try:
            from core.custom_predictor import recognize_segment_custom
            recognize_fn = recognize_segment_custom
        except Exception as e:
            logger.warning(f"自训练模型不可用，回退到 emotion2vec: {e}")
            from core.recognizer import recognize_segment
            recognize_fn = recognize_segment
            use_custom = False
    else:
        from core.recognizer import recognize_segment
        recognize_fn = recognize_segment

    job_dir = jobs_dir / job_id
    segments_dir = job_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    try:
        # Get video duration
        total_duration = get_video_duration(video_path) or 0.0

        # Step 1: Split video by utterances
        job_manager.update_progress(job_id, 0.1, "Splitting video by utterances...")

        def seg_progress(msg):
            job_manager.update_progress(job_id, 0.3, msg)

        segments = split_video_by_utterances(
            video_path, str(segments_dir),
            progress_callback=seg_progress,
        )

        if not segments:
            job_manager.set_failed(job_id, "No segments produced from video")
            return

        # Step 2: Run emotion recognition on each segment
        results = []
        for i, seg in enumerate(segments):
            progress = 0.4 + 0.5 * (i / len(segments))
            job_manager.update_progress(job_id, progress, f"Recognizing emotion {i+1}/{len(segments)}...")

            if use_custom:
                pred = recognize_fn(seg.output_path, text=seg.text)
            else:
                pred = recognize_fn(seg.output_path)

            seg_result = SegmentResult(
                index=seg.index,
                start_time=round(seg.start_time, 2),
                end_time=round(seg.end_time, 2),
                duration=round(seg.end_time - seg.start_time, 2),
                text=seg.text,
                label=pred.get("label_4", 1),
                label_name=pred.get("label_name_4", "calm"),
                label_name_cn=pred.get("label_name_4_cn", "平稳中性"),
                confidence=pred.get("confidence", 0.0),
                emotion2vec_label=pred.get("label_9", 8),
                emotion2vec_label_name=pred.get("label_name_9", "unknown"),
                status="error" if "error" in pred else "ok",
            )
            results.append(seg_result)

        # Save result
        job_result = JobResult(
            job_id=job_id,
            video_name=video_name,
            total_duration=total_duration,
            segments=results,
        )

        # Persist to disk
        result_path = job_dir / "result.json"
        result_path.write_text(job_result.model_dump_json(indent=2), encoding="utf-8")

        job_manager.set_done(job_id, job_result)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        job_manager.set_failed(job_id, str(e))
