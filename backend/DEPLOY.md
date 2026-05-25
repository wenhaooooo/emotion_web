# 后端部署指南

## 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| Python | 3.10+ | 3.10 |
| 内存 | 8 GB | 16 GB+ |
| 磁盘 | 20 GB（模型缓存） | 50 GB+ |
| GPU | 无（CPU 可运行） | NVIDIA GPU + CUDA |
| ffmpeg | 4.0+ | 最新版 |

## 1. 环境准备

### 安装 ffmpeg

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg

# CentOS/RHEL
sudo yum install -y epel-release && sudo yum install -y ffmpeg

# macOS
brew install ffmpeg

# 验证
ffmpeg -version
```

### 安装 Python（推荐 Miniconda）

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建虚拟环境
conda create -n emotion_web python=3.10 -y
conda activate emotion_web
```

## 2. 安装依赖

```bash
cd emotion_web/backend
pip install -r requirements.txt
```

GPU 环境下需额外安装 CUDA 版 PyTorch：

```bash
# CUDA 12.1
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu118
```

## 3. 模型下载

首次运行时模型会自动从 ModelScope 下载，需确保网络通畅。涉及的模型：

| 模型 | 大小 | 用途 |
|------|------|------|
| speech_paraformer-large-vad-punc | ~900 MB | 中文 ASR（VAD + 标点） |
| speech_fsmn_vad | ~10 MB | 语音活动检测 |
| punc_ct-transformer | ~1 GB | 标点恢复 |
| emotion2vec_plus_large | ~1 GB | 情感识别 |

模型缓存目录：`~/.cache/modelscope/hub/models/iic/`

如需离线部署，可提前下载模型到本地，然后修改 `core/asr.py` 和 `core/recognizer.py` 中的 `model_path` 参数指向本地路径。

## 4. 启动服务

### 开发模式

```bash
cd emotion_web/backend
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后监听 `http://0.0.0.0:8000`。

### 生产模式

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **注意：** `--workers` 设为 1，因为 emotion2vec 和 Paraformer 模型加载后会常驻内存，多 worker 会导致重复加载。如需并发，建议通过 Nginx 反向代理 + 多实例方式扩展。

## 5. Systemd 服务（可选）

创建 `/etc/systemd/system/emotion-web.service`：

```ini
[Unit]
Description=Emotion Web Backend
After=network.target

[Service]
Type=simple
User=deploy
WorkingDirectory=/opt/emotion_web/backend
Environment=PATH=/home/deploy/miniconda3/envs/emotion_web/bin:/usr/local/bin:/usr/bin
ExecStart=/home/deploy/miniconda3/envs/emotion_web/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable emotion-web
sudo systemctl start emotion-web
sudo systemctl status emotion-web
```

## 6. Nginx 反向代理（可选）

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 500m;  # 视频上传大小限制

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE 支持
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

## 7. 目录结构

运行时会自动创建以下目录：

```
backend/storage/
├── uploads/     # 用户上传的原始视频
└── jobs/        # 处理任务和结果
    └── {job_id}/
        ├── segments/    # 切分后的视频片段
        └── result.json  # 识别结果
```

## 8. 验证

```bash
# 检查服务健康
curl http://localhost:8000/docs  # Swagger UI

# 上传测试视频
curl -X POST http://localhost:8000/api/upload \
  -F "file=@test_video.mp4"
```

## 9. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 首次启动很慢 | 模型下载中 | 等待下载完成，或提前离线下载 |
| OOM (内存不足) | 模型占用内存大 | 使用 GPU 或增加内存 |
| ffmpeg 报错 | 未安装或版本过低 | `apt install ffmpeg` 或升级 |
| SSE 进度不推送 | Nginx proxy_buffering | 设置 `proxy_buffering off` |
| CUDA 不可用 | PyTorch 版本不匹配 | 重装对应 CUDA 版本的 PyTorch |
