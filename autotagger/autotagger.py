from fastai.vision.all import *
from pandas import DataFrame, read_csv
from fastai.imports import noop
from fastai.callback.progress import ProgressCallback
import gc
import json
import logging
import os
import timm
import torch


def _process_scores(scores, vocab, threshold, limit):
    df = DataFrame({ "tag": vocab, "score": scores })
    df = df[df.score >= threshold].sort_values("score", ascending=False).head(limit)
    return dict(zip(df.tag, df.score))


class Autotagger:
    def __init__(self, model_path="models/model.pth", data_path="test/tags.csv.gz", tags_path="data/tags.json"):
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.use_amp = False
        self.num_workers = 0
        self.batch_size = max(1, int(os.getenv("BATCH_SIZE", "32")))
        self.min_batch_size = max(1, int(os.getenv("MIN_BATCH_SIZE", "1")))
        self.gc_every = max(0, int(os.getenv("GC_EVERY", "0")))
        self.empty_cache_min_images = max(0, int(os.getenv("EMPTY_CACHE_MIN_IMAGES", "0")))
        self.request_count = 0
        self.learn = self.init_model(data_path=data_path, tags_path=tags_path, model_path=model_path)
        logging.info(
            "Autotagger device selected: %s batch_size=%d min_batch_size=%d gc_every=%d empty_cache_min_images=%d",
            self.device.type,
            self.batch_size,
            self.min_batch_size,
            self.gc_every,
            self.empty_cache_min_images,
        )

    def init_model(self, model_path="model/model.pth", data_path="test/tags.csv.gz", tags_path="data/tags.json"):
        df = read_csv(data_path)
        vocab = json.load(open(tags_path))

        dblock = DataBlock(
            blocks=(ImageBlock, MultiCategoryBlock(vocab=vocab)),
            get_x = lambda df: Path("test") / df["filename"],
            get_y = lambda df: df["tags"].split(" "),
            item_tfms = Resize(224, method = ResizeMethod.Squish),
            batch_tfms = [RandomErasing()]
        )

        dls = dblock.dataloaders(df)
        learn = vision_learner(dls, "resnet152", pretrained=False)
        model_file = open(model_path, "rb")
        learn.load(model_file, with_opt=False)
        learn.remove_cb(ProgressCallback)
        learn.logger = noop
        learn.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        # Inference is executed from request threads; keeping DataLoader single-process
        # avoids multiprocessing/resource_sharer instability under concurrent GPU requests.
        self.num_workers = 0
        learn.dls.to(self.device)
        learn.model.to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        return learn

    def _build_test_dl(self, files, bs):
        return self.learn.dls.test_dl(
            files,
            bs=bs,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _run_inference(self, dl):
        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    batch, _ = self.learn.get_preds(dl=dl)
            else:
                batch, _ = self.learn.get_preds(dl=dl)
        return batch

    def _cuda_runtime_error(self, err):
        return self.device.type == "cuda" and "cuda" in str(err).lower()

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
        torch.cuda.empty_cache()
        next_bs = max(self.min_batch_size, bs // 2)
        if next_bs < bs:
            logging.warning("CUDA runtime error detected; retrying with smaller batch size: %d -> %d", bs, next_bs)
            return next_bs
        raise err

    def predict(self, files, threshold=0.01, limit=50, bs=None):
        if not files:
            return []

        if bs is None:
            bs = self.batch_size

        batch = None
        dl = None
        try:
            while True:
                try:
                    dl = self._build_test_dl(files, bs=bs)
                    batch = self._run_inference(dl)
                    break
                except RuntimeError as err:
                    next_bs = self._recover_from_cuda_error(err, bs)
                    if dl is not None:
                        del dl
                        dl = None
                    gc.collect()
                    bs = next_bs

            results = [
                _process_scores(scores, self.learn.dls.vocab, threshold=threshold, limit=limit)
                for scores in batch
            ]
            return results
        finally:
            if batch is not None:
                del batch
            if dl is not None:
                del dl
            self._maybe_run_gc()
            self._maybe_empty_cache(len(files))
