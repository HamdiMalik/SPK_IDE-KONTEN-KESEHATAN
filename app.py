# app.py
# Entry point aplikasi Flask untuk Sistem Pendukung Keputusan (SPK)
# Pembuatan Ide Konten Kesehatan Berbasis Text Mining.
#
# Menginisialisasi server Flask, mengaktifkan CORS untuk integrasi frontend,
# dan mendefinisikan route API utama untuk analisis konten.

import os
import logging
import json
from datetime import datetime

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

from services.dss_engine import generate_content_recommendation
from config.database import close_all_connections

# -------------------------------------------------------------------
# Muat konfigurasi environment dari file .env
# -------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------
# Konfigurasi logging aplikasi
# Format mencakup timestamp, level, nama modul, dan pesan untuk
# kemudahan debugging dan monitoring di lingkungan produksi.
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Inisialisasi Aplikasi Flask
# -------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------
# Konfigurasi CORS (Cross-Origin Resource Sharing)
# Mengizinkan request dari frontend Next.js yang berjalan di port 3000.
# Konfigurasi ini hanya berlaku untuk path /api/* agar lebih aman.
# -------------------------------------------------------------------
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ],
            "methods":  ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)


# -------------------------------------------------------------------
# Custom JSON Encoder
# Menangani serialisasi tipe data yang tidak didukung oleh default
# JSON encoder Flask (seperti numpy types, datetime, dll.)
# -------------------------------------------------------------------
class CustomJSONEncoder(json.JSONEncoder):
    """
    JSON Encoder kustom untuk menangani tipe data Python/NumPy yang tidak
    bisa di-serialize secara default oleh json.dumps() standard.
    """
    def default(self, obj):
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


app.json_encoder = CustomJSONEncoder


# -------------------------------------------------------------------
# ROUTE: Health Check
# Endpoint sederhana untuk memverifikasi bahwa server Flask berjalan.
# Berguna untuk health check Docker container atau load balancer.
# -------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    """
    Endpoint health check untuk memverifikasi status server Flask.

    Returns:
        JSON response dengan status 'ok' dan timestamp saat ini.
        HTTP Status: 200 OK
    """
    return jsonify({
        "status":    "ok",
        "message":   "SPK Konten Kesehatan API berjalan normal.",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version":   "1.0.0",
    }), 200


# -------------------------------------------------------------------
# ROUTE: Analisis Konten (Endpoint Utama)
# -------------------------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Endpoint utama untuk analisis dan rekomendasi ide konten kesehatan.

    Method: POST
    Content-Type: application/json

    Request Body:
        {
            "keyword": "asam lambung"
        }

    Validasi Input:
    - Body request harus berupa JSON yang valid.
    - Field 'keyword' wajib ada dan tidak boleh kosong.
    - Keyword harus berupa string.

    Alur Pemrosesan:
    1. Parse dan validasi JSON request body.
    2. Panggil generate_content_recommendation() dari DSS Engine.
    3. Susun dan kembalikan respons JSON terstruktur.

    Returns:
        JSON Response 200 OK berisi seluruh hasil analisis DSS Engine.
        JSON Response 400 Bad Request jika input tidak valid.
        JSON Response 500 Internal Server Error jika terjadi kegagalan sistem.
    """
    request_start_time = datetime.utcnow()

    # -------------------------------------------------------------------
    # Tahap 1: Parse JSON Request Body
    # -------------------------------------------------------------------
    if not request.is_json:
        logger.warning(
            "Request ke /api/analyze ditolak: Content-Type bukan application/json."
        )
        return jsonify({
            "success": False,
            "error": {
                "code":    "INVALID_CONTENT_TYPE",
                "message": "Request body harus berformat JSON. "
                           "Pastikan header 'Content-Type: application/json' sudah dikirim.",
            },
        }), 400

    try:
        request_data = request.get_json(force=False, silent=False)
    except Exception:
        logger.warning(
            "Request ke /api/analyze ditolak: Body JSON tidak dapat di-parse."
        )
        return jsonify({
            "success": False,
            "error": {
                "code":    "MALFORMED_JSON",
                "message": "Body request tidak dapat di-parse sebagai JSON yang valid.",
            },
        }), 400

    if request_data is None:
        return jsonify({
            "success": False,
            "error": {
                "code":    "EMPTY_REQUEST_BODY",
                "message": "Request body tidak boleh kosong.",
            },
        }), 400

    # -------------------------------------------------------------------
    # Tahap 2: Validasi Field 'keyword'
    # -------------------------------------------------------------------
    keyword = request_data.get("keyword")

    if keyword is None:
        logger.warning("Request /api/analyze ditolak: Field 'keyword' tidak ada.")
        return jsonify({
            "success": False,
            "error": {
                "code":    "MISSING_FIELD",
                "message": "Field 'keyword' wajib disertakan dalam request body.",
            },
        }), 400

    if not isinstance(keyword, str):
        logger.warning(
            "Request /api/analyze ditolak: 'keyword' bukan string (tipe: %s).",
            type(keyword).__name__,
        )
        return jsonify({
            "success": False,
            "error": {
                "code":    "INVALID_FIELD_TYPE",
                "message": "Field 'keyword' harus berupa string.",
            },
        }), 400

    if not keyword.strip():
        logger.warning("Request /api/analyze ditolak: 'keyword' adalah string kosong.")
        return jsonify({
            "success": False,
            "error": {
                "code":    "EMPTY_KEYWORD",
                "message": "Field 'keyword' tidak boleh berupa string kosong atau hanya spasi.",
            },
        }), 400

    # Batasi panjang keyword untuk mencegah abuse
    if len(keyword.strip()) > 100:
        return jsonify({
            "success": False,
            "error": {
                "code":    "KEYWORD_TOO_LONG",
                "message": "Field 'keyword' tidak boleh melebihi 100 karakter.",
            },
        }), 400

    # -------------------------------------------------------------------
    # Tahap 3: Eksekusi DSS Engine
    # -------------------------------------------------------------------
    cleaned_keyword = keyword.strip()
    logger.info(
        "Menerima request analisis untuk keyword='%s' dari %s",
        cleaned_keyword,
        request.remote_addr,
    )

    try:
        analysis_result = generate_content_recommendation(cleaned_keyword)

    except ValueError as val_err:
        # Error ini muncul jika tidak ada data di database untuk keyword yang diberikan
        logger.warning(
            "Analisis keyword='%s' gagal karena ValueError: %s",
            cleaned_keyword,
            str(val_err),
        )
        return jsonify({
            "success": False,
            "error": {
                "code":    "DATA_NOT_FOUND",
                "message": str(val_err),
            },
        }), 404

    except RuntimeError as rt_err:
        # Error ini muncul jika terjadi kegagalan pada komponen sistem (DB, model, ARIMA)
        logger.error(
            "Analisis keyword='%s' gagal karena RuntimeError: %s",
            cleaned_keyword,
            str(rt_err),
            exc_info=True,
        )
        return jsonify({
            "success": False,
            "error": {
                "code":    "SYSTEM_ERROR",
                "message": "Terjadi kegagalan sistem saat memproses analisis. "
                           "Silakan coba kembali dalam beberapa saat.",
                "detail":  str(rt_err),
            },
        }), 500

    except Exception as exc:
        # Tangkap semua exception yang tidak terduga
        logger.critical(
            "Analisis keyword='%s' gagal karena exception tidak terduga: %s",
            cleaned_keyword,
            str(exc),
            exc_info=True,
        )
        return jsonify({
            "success": False,
            "error": {
                "code":    "UNEXPECTED_ERROR",
                "message": "Terjadi kesalahan yang tidak terduga pada server. "
                           "Tim teknis telah diberitahu.",
                "detail":  str(exc),
            },
        }), 500

    # -------------------------------------------------------------------
    # Tahap 4: Susun dan Kembalikan Respons Sukses
    # -------------------------------------------------------------------
    processing_time_ms = round(
        (datetime.utcnow() - request_start_time).total_seconds() * 1000, 2
    )

    response_payload = {
        "success":           True,
        "message":           f"Analisis berhasil diselesaikan untuk keyword '{cleaned_keyword}'.",
        "processing_time_ms": processing_time_ms,
        "data":              analysis_result,
    }

    logger.info(
        "Analisis keyword='%s' berhasil diselesaikan dalam %.2f ms. "
        "Tipe konten: '%s'",
        cleaned_keyword,
        processing_time_ms,
        analysis_result.get("content_recommendation", {}).get("content_type", "N/A"),
    )

    return jsonify(response_payload), 200


# -------------------------------------------------------------------
# Error Handler Global
# Menangani error HTTP standar yang tidak tertangkap oleh route handler.
# -------------------------------------------------------------------
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        "success": False,
        "error": {
            "code":    "ENDPOINT_NOT_FOUND",
            "message": "Endpoint yang diminta tidak ditemukan.",
        },
    }), 404


@app.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify({
        "success": False,
        "error": {
            "code":    "METHOD_NOT_ALLOWED",
            "message": "HTTP method yang digunakan tidak diizinkan untuk endpoint ini.",
        },
    }), 405


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": {
            "code":    "INTERNAL_SERVER_ERROR",
            "message": "Terjadi kesalahan internal pada server.",
        },
    }), 500


# -------------------------------------------------------------------
# Cleanup: Tutup semua koneksi database saat aplikasi dimatikan
# -------------------------------------------------------------------
import atexit

atexit.register(close_all_connections)


# -------------------------------------------------------------------
# Entry Point: Jalankan server Flask
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    flask_env = os.getenv("FLASK_ENV", "production")
    debug_mode = flask_env == "development"

    logger.info(
        "Memulai SPK Konten Kesehatan Flask API di port %d (debug=%s, env=%s)...",
        port,
        debug_mode,
        flask_env,
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode,
        # use_reloader=False karena IndoBERTweet model lazy loading
        # tidak kompatibel dengan Flask's auto-reloader (akan dimuat 2 kali)
        use_reloader=False,
    )
