FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        git \
        gettext-base \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build sqlite-vec from source (pip binary is 32-bit ARM, incompatible with aarch64)
RUN git clone --depth 1 --branch v0.1.6 https://github.com/asg017/sqlite-vec.git /tmp/sqlite-vec && \
    cd /tmp/sqlite-vec && \
    make loadable && \
    mkdir -p /usr/local/lib/sqlite-vec && \
    cp dist/vec0.so /usr/local/lib/sqlite-vec/ && \
    rm -rf /tmp/sqlite-vec

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

RUN mkdir -p /intelligence-data/models

ENV PYTHONUNBUFFERED=1
ENV INTELLIGENCE_DATA_DIR=/intelligence-data
ENV HOMEVAULT_DB_PATH=/data/litloft.db

EXPOSE 8100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
