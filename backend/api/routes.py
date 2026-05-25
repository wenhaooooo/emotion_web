import asyncio
import io
import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse

from api.schemas import JobStatus, JobResult
from tasks.manager import JobManager, run_processing_job

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "storage" / "uploads"
JOBS_DIR = BASE_DIR / "storage" / "jobs"

job_manager = JobManager(JOBS_DIR)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


@router.post("/upload", response_model=JobStatus)
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file format: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Save uploaded file
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    # Create job
    job_id = job_manager.create_job(str(save_path), file.filename)

    # Start background processing
    thread = threading.Thread(
        target=run_processing_job,
        args=(job_manager, job_id, str(save_path), file.filename, JOBS_DIR),
        daemon=True,
    )
    thread.start()

    return job_manager.get_status(job_id)


@router.get("/jobs/{job_id}/progress")
async def job_progress(job_id: str, request: Request):
    status = job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(404, "Job not found")

    if status.status in ("done", "failed"):
        async def single_event():
            yield f"data: {json.dumps(status.model_dump())}\n\n"
        return StreamingResponse(single_event(), media_type="text/event-stream")

    q = job_manager.subscribe_sse(job_id)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.to_thread(q.get, timeout=30)
                    yield f"data: {data}\n\n"
                    parsed = json.loads(data)
                    if parsed.get("status") in ("done", "failed"):
                        break
                except Exception:
                    yield ": keepalive\n\n"
        finally:
            job_manager.cleanup_sse(job_id, q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/jobs/{job_id}/result", response_model=JobResult)
async def job_result(job_id: str):
    result = job_manager.get_result(job_id)
    if result is None:
        status = job_manager.get_status(job_id)
        if status is None:
            raise HTTPException(404, "Job not found")
        if status.status == "failed":
            raise HTTPException(500, f"Job failed: {status.message}")
        raise HTTPException(202, "Job still processing")
    return result


@router.get("/jobs/{job_id}/original-video")
async def job_original_video(job_id: str):
    video_path = job_manager.get_video_path(job_id)
    if video_path is None or not Path(video_path).is_file():
        raise HTTPException(404, "Original video not found")
    return FileResponse(video_path, media_type="video/mp4")


@router.get("/jobs/{job_id}/video/{segment_index}")
async def job_video_segment(job_id: str, segment_index: int):
    segments_dir = JOBS_DIR / job_id / "segments"
    if not segments_dir.exists():
        raise HTTPException(404, "Segments not found")

    matches = list(segments_dir.glob(f"*_seg{segment_index:04d}.mp4"))
    if not matches:
        raise HTTPException(404, f"Segment {segment_index} not found")

    return FileResponse(matches[0], media_type="video/mp4")


@router.get("/jobs/{job_id}/export")
async def job_export_excel(job_id: str):
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

    result = job_manager.get_result(job_id)
    if result is None:
        status = job_manager.get_status(job_id)
        if status is None:
            raise HTTPException(404, "Job not found")
        if status.status == "failed":
            raise HTTPException(500, f"Job failed: {status.message}")
        raise HTTPException(202, "Job still processing")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "情感识别结果"

    # Header
    headers = ["#", "开始时间", "结束时间", "时长(秒)", "文本",
               "emotion2vec标签", "emotion2vec标签名", "教师标签", "教师标签名", "置信度"]
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin_border

    # Data rows
    for seg in result.segments:
        row = [
            seg.index + 1,
            round(seg.start_time, 2),
            round(seg.end_time, 2),
            round(seg.duration, 2),
            seg.text,
            seg.emotion2vec_label,
            seg.emotion2vec_label_name,
            seg.label,
            seg.label_name_cn,
            round(seg.confidence, 4),
        ]
        for col, val in enumerate(row, 1):
            cell = ws.cell(row=seg.index + 2, column=col, value=val)
            cell.border = thin_border

    # Column widths
    widths = [5, 10, 10, 8, 40, 12, 14, 8, 12, 8]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    # Auto-filter
    ws.auto_filter.ref = f"A1:J{len(result.segments) + 1}"

    # Save to buffer
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"{result.video_name}_emotions.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
