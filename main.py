from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from app import app as flask_app

app = FastAPI()

# Mount the Flask app
app.mount("/", WSGIMiddleware(flask_app))

