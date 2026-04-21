# syntax=docker/dockerfile:1.7
FROM golang:1.22-bookworm AS go-builder
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
RUN --mount=type=cache,target=/go/pkg/mod \
  --mount=type=cache,target=/root/.cache/go-build \
  CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /out/autotagger-server ./cmd/server

FROM python:3.14.0-slim AS python-deps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /autotagger
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  UV_LINK_MODE=copy \
  PATH=/opt/venv/bin:/autotagger:$PATH

RUN python -m venv /opt/venv
COPY pyproject.toml uv.lock ./
COPY wheels /wheelhouse
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
  uv sync \
    --frozen \
    --no-dev \
    --no-install-project \
    --python /opt/venv/bin/python \
    --find-links /wheelhouse \
    --index ${PYTORCH_INDEX_URL} \
    --index https://pypi.org/simple

FROM python:3.14.0-slim AS runtime
WORKDIR /autotagger
ARG APP_UID=1000
ARG APP_GID=1000

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PATH=/opt/venv/bin:/autotagger:$PATH

RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends tini curl && \
  rm -rf /var/lib/apt/lists/*

COPY autotag /autotagger/autotag
COPY autotagger /autotagger/autotagger
COPY data /autotagger/data
COPY inference_worker.py /autotagger/inference_worker.py
COPY templates /autotagger/templates
RUN mkdir -p /autotagger/models
COPY --from=python-deps /opt/venv /opt/venv
COPY --from=go-builder /out/autotagger-server /usr/local/bin/autotagger-server

RUN groupadd -g ${APP_GID} appuser || true && \
  useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/sh appuser || true && \
  chown -R ${APP_UID}:${APP_GID} /autotagger
USER ${APP_UID}:${APP_GID}

EXPOSE 5000
ENTRYPOINT ["tini", "--"]
CMD ["autotagger-server"]
