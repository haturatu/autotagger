from fastai.vision.all import *
from pandas import DataFrame, read_csv
from fastai.imports import noop
from fastai.callback.progress import ProgressCallback
from multiprocessing import Pool, cpu_count
from functools import partial
import timm
import sys

def _process_scores(scores, vocab, threshold, limit):
    df = DataFrame({ "tag": vocab, "score": scores })
    df = df[df.score >= threshold].sort_values("score", ascending=False).head(limit)
    return dict(zip(df.tag, df.score))

class Autotagger:
    def __init__(self, model_path="models/model.pth", data_path="test/tags.csv.gz", tags_path="data/tags.json"):
        self.model_path = model_path
        self.learn = self.init_model(data_path=data_path, tags_path=tags_path, model_path=model_path)

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

        if torch.cuda.is_available():
            learn.to_fp16()

        return learn

    def predict(self, files, threshold=0.01, limit=50, bs=64):
        if not files:
            return []

        dl = self.learn.dls.test_dl(files, bs=bs, num_workers=0)
        with torch.inference_mode():
            batch, _ = self.learn.get_preds(dl=dl)

        with Pool(processes=cpu_count()) as pool:
            process_func = partial(_process_scores, vocab=self.learn.dls.vocab, threshold=threshold, limit=limit)
            results = pool.map(process_func, batch)
        
        return results
