# database/models.py
import logging
from typing import List, Dict, Any

from config.database import get_connection, release_connection

logger = logging.getLogger(__name__)


def get_tweets_by_keyword(keyword: str, limit: int = 300) -> List[Dict[str, Any]]:
    """
    Mengambil data tweet dari database berdasarkan pencarian keyword
    pada kolom full_text menggunakan ILIKE (case-insensitive).
    Menggunakan parameter binding (%s) untuk mencegah SQL Injection.
    """
    normalized_keyword = keyword.strip().lower()

    logger.info(
        "Mengambil maksimal %d tweet dengan keyword='%s' dari database.",
        limit,
        normalized_keyword,
    )

    sql_query = """
        SELECT
            full_text,
            normalisasi,
            created_at,
            sentiment
        FROM
            tabel_tweet_mentah
        WHERE
            LOWER(full_text) LIKE %s
            OR LOWER(normalisasi) LIKE %s
        ORDER BY
            created_at DESC
        LIMIT %s;
    """

    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        like_pattern = f"%{normalized_keyword}%"
        cursor.execute(sql_query, (like_pattern, like_pattern, limit))

        rows = cursor.fetchall()

        if not rows:
            logger.warning(
                "Tidak ada data tweet ditemukan untuk keyword='%s'.",
                normalized_keyword,
            )
            return []

        column_names = [description[0] for description in cursor.description]
        result = [dict(zip(column_names, row)) for row in rows]

        logger.info(
            "Berhasil mengambil %d record tweet untuk keyword='%s'.",
            len(result),
            normalized_keyword,
        )

        return result

    except Exception as exc:
        logger.error(
            "Gagal mengeksekusi query untuk keyword='%s': %s",
            normalized_keyword,
            str(exc),
            exc_info=True,
        )
        raise RuntimeError(
            f"Terjadi kesalahan saat mengambil data dari database: {str(exc)}"
        ) from exc

    finally:
        if cursor:
            cursor.close()
        if conn:
            release_connection(conn)
