import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn


CUDA_ERROR_MARKERS = ("cuda", "cublas", "cudnn", "out of memory")
MODEL_NAME = "resnet152"
IMAGE_SIZE = 224
HIDDEN_DIM = 512


def _get_env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        logging.warning("Invalid integer for %s: %r; using default %d", name, raw, default)
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _process_scores(scores, vocab, threshold, limit):
    pairs = [
        (tag, float(score))
        for tag, score in zip(vocab, scores)
        if float(score) >= threshold
    ]
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return dict(pairs[:limit])


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


class TimmFeatureExtractor(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        self.model.reset_classifier(0, "")

    def forward(self, x):
        return self.model.forward_features(x)


class AutotaggerModel(nn.Sequential):
    def __init__(self, model_name: str, num_classes: int):
        backbone = TimmFeatureExtractor(model_name)
        features = backbone.model.num_features
        head = nn.Sequential(
            AdaptiveConcatPool2d(),
            nn.Flatten(),
            nn.BatchNorm1d(features * 2),
            nn.Dropout(p=0.25),
            nn.Linear(features * 2, HIDDEN_DIM, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.Dropout(p=0.5),
            nn.Linear(HIDDEN_DIM, num_classes, bias=False),
        )
        super().__init__(backbone, head)


class Autotagger:
    def __init__(self, model_path="models/model.pth", tags_path="data/tags.json"):
        self.model_path = Path(model_path)
        self.tags_path = Path(tags_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        self.cudnn_benchmark = _get_env_bool("CUDNN_BENCHMARK", True)
        self.batch_size = _get_env_int("BATCH_SIZE", 32, minimum=1)
        self.min_batch_size = _get_env_int("MIN_BATCH_SIZE", 1, minimum=1)
        self.gc_every = _get_env_int("GC_EVERY", 0, minimum=0)
        self.empty_cache_min_images = _get_env_int("EMPTY_CACHE_MIN_IMAGES", 0, minimum=0)
        self.request_count = 0
        self.vocab = self._load_vocab(self.tags_path)
        self.model = self.init_model(self.model_path, len(self.vocab))
        logging.info(
            "Autotagger device=%s amp=%s batch_size=%d min_batch_size=%d cudnn_benchmark=%s gc_every=%d empty_cache_min_images=%d model=%s arch=%s num_classes=%d",
            self.device.type,
            self.use_amp,
            self.batch_size,
            self.min_batch_size,
            self.cudnn_benchmark,
            self.gc_every,
            self.empty_cache_min_images,
            self.model_path,
            MODEL_NAME,
            len(self.vocab),
        )

    def _load_vocab(self, tags_path: Path):
        with tags_path.open("r", encoding="utf-8") as tags_file:
            return json.load(tags_file)

    def _load_checkpoint(self, model_path: Path):
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint format in {model_path}")

        state_dict = {}
        for key, value in checkpoint.items():
            normalized = key
            if normalized.startswith("module."):
                normalized = normalized[len("module."):]
            state_dict[normalized] = value
        return state_dict

    def init_model(self, model_path: Path, num_classes: int):
        model = AutotaggerModel(MODEL_NAME, num_classes)
        state_dict = self._load_checkpoint(model_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"checkpoint incompatibility for {model_path}: missing={missing_keys} unexpected={unexpected_keys}"
            )

        model.eval()
        model.to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
        return model

    def _prepare_image(self, item):
        if isinstance(item, (str, Path)):
            with Image.open(item) as image:
                image = image.convert("RGB")
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
        elif isinstance(item, Image.Image):
            image = item.convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        else:
            raise TypeError(f"Unsupported image input type: {type(item).__name__}")

        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("expected RGB image input")
        return torch.from_numpy(np.transpose(array, (2, 0, 1)))

    def _prepare_batch(self, items):
        tensors = [self._prepare_image(item) for item in items]
        batch = torch.stack(tensors, dim=0)
        if self.device.type == "cuda":
            batch = batch.pin_memory()
            batch = batch.to(self.device, non_blocking=True)
        else:
            batch = batch.to(self.device)
        return batch

    def _run_inference(self, batch):
        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(batch)
            else:
                logits = self.model(batch)
        return torch.sigmoid(logits)

    def _cuda_runtime_error(self, err):
        msg = str(err).lower()
        return self.device.type == "cuda" and any(marker in msg for marker in CUDA_ERROR_MARKERS)

    def _maybe_run_gc(self):
        self.request_count += 1
        if self.gc_every > 0 and self.request_count % self.gc_every == 0:
            gc.collect()

    def _maybe_empty_cache(self, image_count):
        if self.device.type != "cuda":
            return
        if self.empty_cache_min_images > 0 and image_count >= self.empty_cache_min_images:
            torch.cuda.empty_cache()

    def _recover_from_cuda_error(self, err, bs):
        if not self._cuda_runtime_error(err):
            raise err

        msg = str(err).lower()
        logging.warning("CUDA inference error at batch_size=%d: %s", bs, msg)
        torch.cuda.empty_cache()
        next_bs = max(self.min_batch_size, bs // 2)
        if next_bs < bs:
            logging.warning("Retrying with smaller batch size: %d -> %d", bs, next_bs)
            return next_bs
        raise err

    def predict(self, files, threshold=0.01, limit=50, bs=None):
        if not files:
            return []

        if bs is None:
            bs = self.batch_size

        outputs = []
        batch = None
        scores = None
        try:
            start = 0
            while start < len(files):
                current_bs = min(bs, len(files) - start)
                while True:
                    try:
                        batch_items = files[start : start + current_bs]
                        batch = self._prepare_batch(batch_items)
                        scores = self._run_inference(batch).detach().cpu().numpy()
                        outputs.extend(
                            _process_scores(score_row, self.vocab, threshold=threshold, limit=limit)
                            for score_row in scores
                        )
                        start += current_bs
                        break
                    except RuntimeError as err:
                        next_bs = self._recover_from_cuda_error(err, current_bs)
                        if batch is not None:
                            del batch
                            batch = None
                        if scores is not None:
                            del scores
                            scores = None
                        gc.collect()
                        current_bs = next_bs
                if batch is not None:
                    del batch
                    batch = None
                if scores is not None:
                    del scores
                    scores = None
            return outputs
        finally:
            if batch is not None:
                del batch
            if scores is not None:
                del scores
            self._maybe_run_gc()
            self._maybe_empty_cache(len(files))
