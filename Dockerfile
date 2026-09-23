# 抗裁切水印实验台 —— 运行镜像
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OpenCV/Pillow 运行所需的最小系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# 通过环境变量配置：HOST/PORT/WORKERS
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1

# 亚像素网格搜索为 CPU 密集型，多 worker 可并行检测；默认 1 个即可使用
CMD uvicorn app.main:app --host ${HOST} --port ${PORT} --workers ${WORKERS}
