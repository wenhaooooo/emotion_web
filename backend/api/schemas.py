from pydantic import BaseModel
from typing import Optional


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending / processing / done / failed
    progress: float = 0.0  # 0.0 ~ 1.0
    message: str = ""


class SegmentResult(BaseModel):
    index: int
    start_time: float
    end_time: float
    duration: float
    text: str
    label: int  # 0-3
    label_name: str
    label_name_cn: str
    confidence: float
    emotion2vec_label: int
    emotion2vec_label_name: str
    status: str = "ok"  # ok / error


class JobResult(BaseModel):
    job_id: str
    video_name: str
    total_duration: float
    segments: list[SegmentResult]
