from fastai.vision.all import *
from pandas import DataFrame, read_csv
from fastai.imports import noop
from fastai.callback.progress import ProgressCallback
from multiprocessing import cpu_count
import logging
import timm
import sys
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
        self.learn = self.init_model(data_path=data_path, tags_path=tags_path, model_path=model_path)
        logging.info("Autotagger device selected: %s", self.device.type)

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

    def predict(self, files, threshold=0.01, limit=50, bs=64):
        if not files:
            return []

        dl = self.learn.dls.test_dl(
            files,
            bs=bs,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        try:
            with torch.inference_mode():
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        batch, _ = self.learn.get_preds(dl=dl)
                else:
                    batch, _ = self.learn.get_preds(dl=dl)
        except RuntimeError as err:
            # If CUDA fails at runtime, retry once on CPU.
            if self.device.type != "cuda" or "cuda" not in str(err).lower():
                raise
            self.device = torch.device("cpu")
            self.use_amp = False
            self.num_workers = 0
            self.learn.dls.to(self.device)
            self.learn.model.to(self.device)
            logging.warning("CUDA runtime error detected; falling back to CPU for inference.")
            dl = self.learn.dls.test_dl(files, bs=bs, num_workers=0, pin_memory=False)
            with torch.inference_mode():
                batch, _ = self.learn.get_preds(dl=dl)

        results = [
            _process_scores(scores, self.learn.dls.vocab, threshold=threshold, limit=limit)
            for scores in batch
        ]

        return results
