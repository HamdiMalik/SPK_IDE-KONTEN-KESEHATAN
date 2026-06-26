# services/indobert_service.py
# Modul layanan inferensi sentimen menggunakan model IndoBERTweet.
# Model dimuat dari direktori lokal ./saved_models/indobert_sentiment.
# Menggunakan strategi batching dan torch.no_grad() untuk efisiensi latensi.

import os
import logging
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

_MODEL_LOCAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "saved_models",
    "indobert_sentiment",
)

_LABEL_MAP = {
    0: "negatif",
    1: "netral",
    2: "positif",
}

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None


def _load_model_and_tokenizer() -> None:
    global _tokenizer, _model

    if not os.path.isdir(_MODEL_LOCAL_PATH):
        raise FileNotFoundError(
            f"Direktori model IndoBERTweet tidak ditemukan di: '{_MODEL_LOCAL_PATH}'. "
            "Pastikan model sudah di-fine-tune dan disimpan ke direktori tersebut "
            "menggunakan model.save_pretrained() dan tokenizer.save_pretrained()."
        )

    logger.info("Memuat tokenizer IndoBERTweet dari path lokal: %s", _MODEL_LOCAL_PATH)
    _tokenizer = AutoTokenizer.from_pretrained(
        _MODEL_LOCAL_PATH,
        local_files_only=True,
    )

    logger.info("Memuat model IndoBERTweet dari path lokal: %s", _MODEL_LOCAL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_LOCAL_PATH,
        local_files_only=True,
        use_safetensors=True,
    )

    _model.eval()

    logger.info("Model IndoBERTweet berhasil dimuat dan dikunci ke mode evaluasi (eval mode).")


def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _load_model_and_tokenizer()
    return _tokenizer


def get_model() -> AutoModelForSequenceClassification:
    global _model
    if _model is None:
        _load_model_and_tokenizer()
    return _model


def predict_sentiment_batch(clean_texts: List[str]) -> List[int]:
    if not clean_texts:
        logger.warning("predict_sentiment_batch dipanggil dengan list teks kosong.")
        return []

    tokenizer = get_tokenizer()
    model = get_model()

    processed_texts = [text if text.strip() else "[PAD]" for text in clean_texts]

    logger.info("Memulai inferensi batch IndoBERTweet untuk %d teks...", len(processed_texts))

    # Proses dalam mini-batch 32 untuk menghindari OOM (Out of Memory)
    # jika data terlalu besar untuk diproses sekaligus
    batch_size = 32
    all_predictions = []

    for i in range(0, len(processed_texts), batch_size):
        mini_batch = processed_texts[i : i + batch_size]

        encoded_inputs = tokenizer(
            mini_batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**encoded_inputs)

        batch_predictions = torch.argmax(outputs.logits, dim=1).tolist()
        all_predictions.extend(batch_predictions)

        logger.info(
            "Mini-batch %d/%d selesai (%d teks).",
            (i // batch_size) + 1,
            (len(processed_texts) + batch_size - 1) // batch_size,
            len(mini_batch),
        )

    logger.info(
        "Inferensi batch selesai. Distribusi prediksi — "
        "Negatif: %d, Netral: %d, Positif: %d",
        all_predictions.count(0),
        all_predictions.count(1),
        all_predictions.count(2),
    )

    return all_predictions


def get_label_name(label_index: int) -> str:
    return _LABEL_MAP.get(label_index, "tidak diketahui")
