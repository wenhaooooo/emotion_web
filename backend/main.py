import os
import logging
import warnings
from pathlib import Path
from contextlib import asynccontextmanager

# 抑制第三方库的无关警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 使用 HuggingFace 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 配置日志：确保所有模块的 INFO 日志都能输出到终端
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    force=True,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.face_recognizer import init_face_recognizer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
JOBS_DIR = STORAGE_DIR / "jobs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    # 同步加载面部表情识别模型
    logger.info("正在加载面部表情识别模型...")
    try:
        init_face_recognizer()
        logger.info("面部表情识别模型加载完成")
    except Exception as e:
        logger.error(f"面部表情识别模型加载失败: {e}")

    # 预加载自训练情感识别模型（优先），回退到 emotion2vec
    import os
    if os.environ.get("EMOTION_MODEL", "custom") == "custom":
        logger.info("正在加载自训练多模态情感识别模型...")
        try:
            from core.custom_predictor import _get_custom_predictor
            predictor = _get_custom_predictor()
            predictor._ensure_model()  # 触发实际模型加载
            logger.info("自训练多模态情感识别模型加载完成")
        except Exception as e:
            logger.error(f"自训练模型加载失败: {e}")
            logger.info("尝试加载 emotion2vec 备用模型...")
            try:
                from core.recognizer import _get_predictor
                _get_predictor()
                logger.info("emotion2vec 备用模型加载完成")
            except Exception as e2:
                logger.error(f"emotion2vec 备用模型加载也失败: {e2}")
    else:
        logger.info("正在加载语音情感识别模型 (emotion2vec)...")
        try:
            from core.recognizer import _get_predictor
            _get_predictor()
            logger.info("语音情感识别模型加载完成")
        except Exception as e:
            logger.error(f"语音情感识别模型加载失败: {e}")

    yield


app = FastAPI(title="Emotion Web", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
