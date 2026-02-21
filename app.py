#!/usr/bin/env python

from os import getenv
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from dotenv import load_dotenv
from autotagger import Autotagger
from base64 import b64encode
from fastai.vision.core import PILImage
from flask import Flask, request, render_template, jsonify, abort
from werkzeug.exceptions import HTTPException
import torch
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
model_path = getenv("MODEL_PATH", "models/model.pth")
gpu_parallelism = max(1, int(getenv("GPU_PARALLELISM", "4")))


class AutotaggerRunner:
    def __init__(self, model_path, gpu_parallelism):
        probe = Autotagger(model_path)
        self.is_gpu = probe.device.type == "cuda"
        self._lock = Lock()
        self._index = 0

        if self.is_gpu:
            self._taggers = [probe] + [Autotagger(model_path) for _ in range(gpu_parallelism - 1)]
            self._executor = ThreadPoolExecutor(max_workers=len(self._taggers), thread_name_prefix="gpu-infer")
            logging.info("GPU inference pool enabled with %d workers.", len(self._taggers))
        else:
            self._taggers = [probe]
            self._executor = None
            logging.info("CPU inference mode enabled (single worker).")

    def _next_tagger(self):
        with self._lock:
            tagger = self._taggers[self._index]
            self._index = (self._index + 1) % len(self._taggers)
            return tagger

    def predict(self, images, threshold, limit):
        tagger = self._next_tagger()
        if not self.is_gpu:
            return tagger.predict(images, threshold=threshold, limit=limit)

        future = self._executor.submit(tagger.predict, images, threshold, limit)
        return future.result()


autotagger = AutotaggerRunner(model_path, gpu_parallelism)

# This is necessary for Gunicorn to work with multiple workers and preloading enabled.
torch.set_num_threads(1)
if not autotagger.is_gpu:
    autotagger._taggers[0].learn.model.share_memory()

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["JSON_PRETTYPRINT_REGULAR"] = True

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/evaluate", methods=["POST"])
def evaluate():
    files = request.files.getlist("file")
    threshold = float(request.values.get("threshold", 0.1))
    output = request.values.get("format", "html")
    limit = int(request.values.get("limit", 50))

    images = [PILImage.create(file) for file in files]
    predictions = autotagger.predict(images, threshold=threshold, limit=limit)

    if output == "html":
        for file in files:
            file.seek(0)

        base64data = [b64encode(file.read()).decode() for file in files]
        return render_template("evaluate.html", predictions=zip(base64data, predictions))
    elif output == "json":
        predictions = [{ "filename": file.filename, "tags": tags } for file, tags in zip(files, predictions)]
        return jsonify(predictions)
    else:
        abort(400)

@app.errorhandler(HTTPException)
def handle_http_exception(exception):
    output = request.values.get("format", "html")

    if hasattr(exception, "original_exception"):
        error = exception.original_exception.__class__.__name__
        message = str(exception.original_exception)
    else:
        error = exception.__class__.__name__
        message = str(exception)

    if output == "html":
        return render_template("error.html", error=error, message=message)
    else:
        return jsonify({ "error": error, "message": message }), exception.code

if __name__ == "__main__":
    app.run(host="0.0.0.0")
