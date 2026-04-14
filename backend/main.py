import os
import pickle
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


MODEL_STORE: dict[str, Any] = {}
MIN_RELIABLE_TOKENS = 3


def _allowed_origins() -> list[str]:
    frontend_url = os.environ.get("FRONTEND_URL", "").strip()
    if not frontend_url:
        return ["http://localhost:3000"]
    origins = [origin.strip() for origin in frontend_url.split(",") if origin.strip()]
    return origins or ["http://localhost:3000"]


def _models_dir() -> str:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("MODELS_DIR", os.path.abspath(os.path.join(backend_dir, "..", "models")))


def _safe_accuracy(metrics: dict[str, Any], key: str) -> float | None:
    entry = metrics.get(key, {})
    value = entry.get("accuracy") if isinstance(entry, dict) else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_threshold(metrics: dict[str, Any], key: str) -> float:
    entry = metrics.get(key, {})
    value = entry.get("threshold") if isinstance(entry, dict) else None
    if isinstance(value, (int, float)):
        return float(max(0.0, min(1.0, value)))
    return 0.5


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def load_models() -> None:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model

    models_dir = _models_dir()
    required = [
        "nb_model.pkl",
        "svm_model.pkl",
        "tokenizer.pkl",
        "metrics.pkl",
        "lstm_model.h5",
        "cnn_lstm_model.h5",
    ]

    missing = [name for name in required if not os.path.exists(os.path.join(models_dir, name))]
    if missing:
        raise FileNotFoundError(
            "Missing model files in MODELS_DIR. Missing: " + ", ".join(missing)
        )

    with open(os.path.join(models_dir, "nb_model.pkl"), "rb") as file:
        MODEL_STORE["nb"] = pickle.load(file)

    with open(os.path.join(models_dir, "svm_model.pkl"), "rb") as file:
        MODEL_STORE["svm"] = pickle.load(file)

    with open(os.path.join(models_dir, "tokenizer.pkl"), "rb") as file:
        MODEL_STORE["tokenizer"] = pickle.load(file)

    with open(os.path.join(models_dir, "metrics.pkl"), "rb") as file:
        MODEL_STORE["metrics"] = pickle.load(file)

    MODEL_STORE["lstm"] = load_model(os.path.join(models_dir, "lstm_model.h5"), compile=False)
    MODEL_STORE["cnn_lstm"] = load_model(os.path.join(models_dir, "cnn_lstm_model.h5"), compile=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()
    yield


app = FastAPI(title="Sentiment Flow ML Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


def ml_predict(model: Any, text: str) -> tuple[float, int]:
    start = time.perf_counter()
    probabilities = model.predict_proba([text])[0]
    latency_ms = int((time.perf_counter() - start) * 1000)
    return float(probabilities[1]), latency_ms


def dl_predict(model: Any, text: str, tokenizer: Any, max_len: int = 200) -> tuple[float, int]:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    start = time.perf_counter()
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post")
    probability = float(model.predict(padded, verbose=0)[0][0])
    latency_ms = int((time.perf_counter() - start) * 1000)
    return probability, latency_ms


def _model_output(
    *,
    key: str,
    name: str,
    family: str,
    pos_prob: float,
    latency_ms: int,
    threshold: float,
    test_accuracy: float | None,
    short_text_guard: bool,
) -> dict[str, Any]:
    label = "positive" if pos_prob >= threshold else "negative"
    confidence = _clamp01(abs(pos_prob - threshold) * 2)

    note = f"Calibrated threshold {threshold:.2f}."
    if short_text_guard:
        confidence = _clamp01(confidence * 0.5)
        note = f"Short-text guard active (<{MIN_RELIABLE_TOKENS} tokens). {note}"

    return {
        "key": key,
        "name": name,
        "family": family,
        "pos_prob": _clamp01(pos_prob),
        "latency_ms": latency_ms,
        "test_accuracy": test_accuracy,
        "threshold": threshold,
        "label": label,
        "confidence": confidence,
        "note": note,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_dir": _models_dir(),
        "loaded_models": list(MODEL_STORE.keys()),
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")

    if not MODEL_STORE:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    start_total = time.perf_counter()
    token_count = _token_count(text)
    short_text_guard = token_count < MIN_RELIABLE_TOKENS

    nb_prob, nb_ms = ml_predict(MODEL_STORE["nb"], text)
    svm_prob, svm_ms = ml_predict(MODEL_STORE["svm"], text)
    lstm_prob, lstm_ms = dl_predict(MODEL_STORE["lstm"], text, MODEL_STORE["tokenizer"])
    cnn_prob, cnn_ms = dl_predict(MODEL_STORE["cnn_lstm"], text, MODEL_STORE["tokenizer"])

    metrics = MODEL_STORE.get("metrics", {})

    models = [
        _model_output(
            key="naive_bayes",
            name="Naive Bayes",
            family="Classical ML",
            pos_prob=nb_prob,
            latency_ms=nb_ms,
            threshold=_safe_threshold(metrics, "Naive Bayes"),
            test_accuracy=_safe_accuracy(metrics, "Naive Bayes"),
            short_text_guard=short_text_guard,
        ),
        _model_output(
            key="svm",
            name="SVM",
            family="Classical ML",
            pos_prob=svm_prob,
            latency_ms=svm_ms,
            threshold=_safe_threshold(metrics, "SVM"),
            test_accuracy=_safe_accuracy(metrics, "SVM"),
            short_text_guard=short_text_guard,
        ),
        _model_output(
            key="lstm",
            name="LSTM",
            family="Deep Learning",
            pos_prob=lstm_prob,
            latency_ms=lstm_ms,
            threshold=_safe_threshold(metrics, "LSTM"),
            test_accuracy=_safe_accuracy(metrics, "LSTM"),
            short_text_guard=short_text_guard,
        ),
        _model_output(
            key="cnn_lstm",
            name="CNN-LSTM",
            family="Hybrid DL",
            pos_prob=cnn_prob,
            latency_ms=cnn_ms,
            threshold=_safe_threshold(metrics, "CNN-LSTM"),
            test_accuracy=_safe_accuracy(metrics, "CNN-LSTM"),
            short_text_guard=short_text_guard,
        ),
    ]

    pos_votes = sum(1 for item in models if item["label"] == "positive")
    total_votes = len(models)

    if short_text_guard:
        consensus_label = "indeterminate"
        consensus_reason = f"Input is too short ({token_count} tokens)."
    elif total_votes == 4 and pos_votes == 2:
        consensus_label = "indeterminate"
        consensus_reason = "Exact 2/4 split."
    else:
        consensus_label = "positive" if pos_votes > total_votes / 2 else "negative"
        consensus_reason = "Majority decision from calibrated model thresholds."

    avg_confidence = _clamp01(sum(float(item["confidence"]) for item in models) / max(total_votes, 1))

    return {
        "models": models,
        "consensus": {
            "label": consensus_label,
            "pos_votes": pos_votes,
            "total": total_votes,
            "confidence": avg_confidence,
            "reason": consensus_reason,
        },
        "meta": {
            "latency_ms": int((time.perf_counter() - start_total) * 1000),
            "token_count": token_count,
            "short_text_guard": short_text_guard,
            "guard_message": (
                f"Short-text guard active: at least {MIN_RELIABLE_TOKENS} tokens recommended."
                if short_text_guard
                else ""
            ),
        },
    }
