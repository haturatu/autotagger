#!/usr/bin/env python3

import json
import logging
import os
import sys
from pathlib import Path

from fastai.vision.core import PILImage

from autotagger import Autotagger


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_tagger() -> Autotagger:
    model_path = os.getenv("MODEL_PATH", "models/model.pth")
    return Autotagger(model_path)


def predict_files(tagger: Autotagger, files: list[str], threshold: float, limit: int):
    images = []
    names = []
    for path in files:
        with open(path, "rb") as f:
            images.append(PILImage.create(f))
        names.append(Path(path).name)

    predictions = tagger.predict(images, threshold=threshold, limit=limit)
    return [{"filename": name, "tags": tags} for name, tags in zip(names, predictions)]


def main() -> int:
    tagger = build_tagger()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            files = req.get("files", [])
            threshold = float(req.get("threshold", 0.1))
            limit = int(req.get("limit", 50))

            predictions = predict_files(tagger, files, threshold, limit)
            res = {"id": req_id, "predictions": predictions}
        except Exception as e:
            res = {"id": req_id, "error": f"{type(e).__name__}: {e}"}

        sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
