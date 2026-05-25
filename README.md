# Emotion Web

教师情感识别 Web 应用。上传教学视频，自动按句子切分，使用 FunASR Paraformer 进行中文语音识别，emotion2vec+ 进行情感分类，展示时间轴和结果表格。

## 功能

- 上传教学视频（MP4 等格式）
- FunASR Paraformer 中文 ASR（带 VAD + 标点），按句号/问号/感叹号自动切分
- emotion2vec+ 情感识别，支持 9 类（原始）和 4 类（教学场景）标签
- 前端时间轴可视化、视频播放器、结果表格（可按情感筛选）
- 导出 Excel 报告

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 语音识别 | FunASR Paraformer large（VAD + 标点） |
| 情感识别 | emotion2vec+ large |
| 音视频处理 | ffmpeg + ffmpeg-python |
| 前端框架 | Vue 3 + Vite |
| HTTP 客户端 | axios |

## 情感标签

### 4 类标签（教学场景）

| 标签 | 中文 | 说明 |
|------|------|------|
| 0 - enthusiastic | 热情投入 | 积极、兴奋 |
| 1 - calm | 平稳中性 | 中性、平稳 |
| 2 - negative | 消极低落 | 消极、悲伤 |
| 3 - tense | 紧张焦虑 | 紧张、焦虑 |

### 9 类标签（emotion2vec 原始）

angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
# Server at http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# UI at http://localhost:5173
```

## 项目结构

```
emotion_web/
├── README.md
├── backend/
│   ├── main.py                  # FastAPI 入口
│   ├── requirements.txt
│   ├── api/
│   │   ├── routes.py            # REST API 路由
│   │   └── schemas.py           # Pydantic 数据模型
│   ├── core/
│   │   ├── asr.py               # FunASR Paraformer ASR
│   │   ├── segmentation.py      # 视频按句子切分
│   │   ├── recognizer.py        # emotion2vec 情感识别
│   │   └── ffmpeg_utils.py      # ffmpeg 音视频工具
│   ├── tasks/
│   │   └── manager.py           # 后台任务管理
│   └── storage/                 # 上传文件和处理结果
└── frontend/
    ├── src/
    │   ├── App.vue
    │   ├── api.js               # HTTP 和 SSE 客户端
    │   ├── emotionConfig.js     # 情感标签配置
    │   └── components/
    │       ├── VideoUpload.vue   # 文件上传
    │       ├── ProgressPanel.vue # 处理进度
    │       ├── VideoPlayer.vue   # 视频播放器 + 时间轴
    │       ├── EmotionTimeline.vue # 情感时间轴
    │       └── ResultTable.vue   # 结果表格
    └── package.json
```
