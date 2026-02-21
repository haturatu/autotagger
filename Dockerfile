FROM python:3.12.3-slim
WORKDIR /autotagger

ENV MPLCONFIGDIR=/tmp/matplotlib

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PATH=/autotagger:$PATH

RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends tini aria2 && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN \
  pip install -r requirements.txt

RUN \
  mkdir models && \
  aria2c \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --continue=true \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --file-allocation=none \
    --dir=models \
    --out=model.pth \
    https://github.com/danbooru/autotagger/releases/download/2022.06.20-233624-utc/model.pth

COPY . .
RUN getent group nobody || groupadd nobody
RUN chown -R nobody:nobody /autotagger
USER nobody

EXPOSE 5000
ENTRYPOINT ["tini", "--"]
#CMD ["autotag"]
#CMD ["flask", "run", "--host", "0.0.0.0"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
