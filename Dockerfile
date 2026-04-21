# syntax=docker/dockerfile:1.7
FROM golang:1.22-bookworm AS go-builder
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
RUN --mount=type=cache,target=/go/pkg/mod \
  --mount=type=cache,target=/root/.cache/go-build \
  CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /out/autotagger-server ./cmd/server

FROM python:3.12.3-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /autotagger
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PATH=/opt/venv/bin:/autotagger:$PATH

RUN python -m venv /opt/venv
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install \
    --python /opt/venv/bin/python \
    --index-url ${PYTORCH_INDEX_URL} \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

FROM python:3.12.3-slim AS runtime
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

COPY . .
RUN mkdir -p /autotagger/models
COPY --from=builder /opt/venv /opt/venv
COPY --from=go-builder /out/autotagger-server /usr/local/bin/autotagger-server

RUN groupadd -g ${APP_GID} appuser || true && \
  useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/sh appuser || true && \
  chown -R ${APP_UID}:${APP_GID} /autotagger
USER ${APP_UID}:${APP_GID}

EXPOSE 5000
ENTRYPOINT ["tini", "--"]
CMD ["autotagger-server"]
