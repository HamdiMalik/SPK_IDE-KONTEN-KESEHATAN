# config/database.py
# Modul konfigurasi dan manajemen koneksi database PostgreSQL.
# Menggunakan psycopg2 connection pool untuk efisiensi dan keamanan koneksi.

import os
import logging
from urllib.parse import urlparse

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

# Muat konfigurasi dari file .env
load_dotenv()

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Parsing DATABASE_URL menjadi parameter koneksi individual.
# Ini memungkinkan fleksibilitas konfigurasi baik via URL maupun
# environment variable terpisah.
# -------------------------------------------------------------------

def _parse_database_url(url: str) -> dict:
    """
    Parse DATABASE_URL string ke dalam dictionary parameter koneksi psycopg2.

    Args:
        url (str): URL koneksi PostgreSQL dengan format
                   postgresql://user:password@host:port/dbname

    Returns:
        dict: Dictionary berisi parameter koneksi psycopg2.

    Raises:
        ValueError: Jika format URL tidak valid atau skema bukan postgresql.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("postgresql", "postgres"):
        raise ValueError(
            f"Skema URL database tidak valid: '{parsed.scheme}'. "
            "Harus menggunakan 'postgresql' atau 'postgres'."
        )

    if not parsed.hostname:
        raise ValueError("Host database tidak ditemukan dalam DATABASE_URL.")

    if not parsed.path or parsed.path == "/":
        raise ValueError("Nama database tidak ditemukan dalam DATABASE_URL.")

    return {
        "host":     parsed.hostname,
        "port":     parsed.port or 5432,
        "dbname":   parsed.path.lstrip("/"),
        "user":     parsed.username,
        "password": parsed.password,
    }


# -------------------------------------------------------------------
# Connection Pool — dibuat sekali saat modul pertama kali di-import.
# ThreadedConnectionPool aman digunakan dalam lingkungan multi-threaded
# seperti server Flask yang melayani banyak request bersamaan.
# -------------------------------------------------------------------

_connection_pool: pool.ThreadedConnectionPool | None = None


def _create_pool() -> pool.ThreadedConnectionPool:
    """
    Membuat instance ThreadedConnectionPool berdasarkan DATABASE_URL dari .env.

    Returns:
        pool.ThreadedConnectionPool: Instance connection pool yang siap digunakan.

    Raises:
        EnvironmentError: Jika DATABASE_URL tidak ditemukan di environment.
        psycopg2.OperationalError: Jika koneksi ke database gagal.
    """
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise EnvironmentError(
            "Environment variable 'DATABASE_URL' tidak ditemukan. "
            "Pastikan file .env sudah dikonfigurasi dengan benar."
        )

    conn_params = _parse_database_url(database_url)

    logger.info(
        "Membuat connection pool ke database '%s' di host '%s:%s'...",
        conn_params["dbname"],
        conn_params["host"],
        conn_params["port"],
    )

    created_pool = pool.ThreadedConnectionPool(
        minconn=2,   # Jumlah koneksi minimum yang selalu terbuka
        maxconn=10,  # Jumlah koneksi maksimum yang bisa dibuat sekaligus
        **conn_params,
    )

    logger.info("Connection pool berhasil dibuat.")
    return created_pool


def get_pool() -> pool.ThreadedConnectionPool:
    """
    Mengembalikan instance connection pool global (singleton).
    Pool dibuat hanya sekali saat pertama kali fungsi ini dipanggil.

    Returns:
        pool.ThreadedConnectionPool: Instance connection pool aktif.
    """
    global _connection_pool
    if _connection_pool is None or _connection_pool.closed:
        _connection_pool = _create_pool()
    return _connection_pool


def get_connection() -> psycopg2.extensions.connection:
    """
    Mengambil sebuah koneksi dari pool.
    Koneksi yang didapat HARUS dikembalikan ke pool setelah selesai digunakan
    dengan memanggil fungsi release_connection().

    Returns:
        psycopg2.extensions.connection: Objek koneksi database aktif.
    """
    return get_pool().getconn()


def release_connection(conn: psycopg2.extensions.connection) -> None:
    """
    Mengembalikan koneksi yang sudah selesai digunakan kembali ke pool.

    Args:
        conn (psycopg2.extensions.connection): Koneksi yang akan dikembalikan.
    """
    if conn:
        get_pool().putconn(conn)


def close_all_connections() -> None:
    """
    Menutup semua koneksi dalam pool. Dipanggil saat aplikasi Flask shutdown.
    """
    global _connection_pool
    if _connection_pool and not _connection_pool.closed:
        _connection_pool.closeall()
        logger.info("Semua koneksi dalam pool telah ditutup.")
        _connection_pool = None
