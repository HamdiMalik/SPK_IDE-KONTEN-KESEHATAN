# services/dss_engine.py
import logging
import collections
from typing import Dict, Any, List

from database.models import get_tweets_by_keyword
from services.nlp_pipeline import clean_text
from services.indobert_service import predict_sentiment_batch, get_label_name
from services.arima_service import forecast_trend

logger = logging.getLogger(__name__)

_SENTIMENT_LABELS = {
    0: "negatif",
    1: "netral",
    2: "positif",
}


def _calculate_sentiment_distribution(sentiment_labels: List[int]) -> Dict[str, Any]:
    total = len(sentiment_labels)
    counter = collections.Counter(sentiment_labels)

    counts = {
        "negatif": counter.get(0, 0),
        "netral":  counter.get(1, 0),
        "positif": counter.get(2, 0),
    }

    percentages = {
        label: round((count / total) * 100, 2)
        for label, count in counts.items()
    }

    dominant_int = max(counter, key=counter.get) if counter else 1
    dominant_name = _SENTIMENT_LABELS.get(dominant_int, "netral")

    return {
        "counts":       counts,
        "percentages":  percentages,
        "dominant":     dominant_name,
        "dominant_idx": dominant_int,
    }


def _extract_top_keywords(
    clean_texts: List[str],
    dominant_label_idx: int,
    all_labels: List[int],
    top_n: int = 2,
) -> List[str]:
    dominant_texts = [
        text
        for text, label in zip(clean_texts, all_labels)
        if label == dominant_label_idx and text.strip()
    ]

    if not dominant_texts:
        logger.warning("Tidak ada teks dari kelas dominan untuk ekstraksi keyword.")
        return []

    all_words = " ".join(dominant_texts).split()
    meaningful_words = [word for word in all_words if len(word) >= 4]

    if not meaningful_words:
        return all_words[:top_n] if all_words else []

    word_counter = collections.Counter(meaningful_words)
    top_keywords = [word for word, _ in word_counter.most_common(top_n)]

    logger.info("Top %d kata kunci dari kelas dominan: %s", top_n, top_keywords)
    return top_keywords


def _generate_recommendation_text(
    dominant_sentiment: str,
    trend_status: str,
    top_keywords: List[str],
    keyword: str,
) -> Dict[str, str]:
    kw1 = top_keywords[0] if len(top_keywords) > 0 else keyword
    kw2 = top_keywords[1] if len(top_keywords) > 1 else "gejala"
    topic = keyword.title()

    if dominant_sentiment == "negatif":
        if trend_status == "Naik":
            return {
                "content_type": "Edukasi Solutif (Urgent)",
                "recommendation": (
                    f"Perbincangan negatif tentang '{topic}' sedang meningkat tajam, "
                    f"terutama seputar topik '{kw1}' dan '{kw2}'. Ini adalah momen kritis "
                    f"untuk mempublikasikan konten edukatif yang memberikan solusi cepat dan "
                    f"praktis. Buat konten berformat carousel atau video pendek yang secara "
                    f"langsung menjawab keluhan '{kw1}' dengan panduan pertolongan pertama "
                    f"atau langkah segera yang bisa dilakukan audiens saat mengalami gejala '{kw2}'. "
                    f"Sertakan rekomendasi kapan harus segera berkonsultasi ke dokter."
                ),
                "content_pillars": (
                    f"1. Pertolongan pertama saat {kw1} menyerang\n"
                    f"2. Makanan/minuman yang aman dikonsumsi saat {kw2}\n"
                    f"3. Tanda bahaya {topic} yang tidak boleh diabaikan\n"
                    f"4. Panduan konsultasi dokter yang tepat waktu"
                ),
                "call_to_action": (
                    f"Posting SEGERA dalam 24 jam ke depan. Gunakan hashtag trending "
                    f"terkait {topic} dan tag akun kesehatan terpercaya untuk amplifikasi."
                ),
            }
        elif trend_status == "Turun":
            return {
                "content_type": "Edukasi Pencegahan",
                "recommendation": (
                    f"Meskipun sentimen negatif masih dominan untuk topik '{topic}', "
                    f"tren menunjukkan penurunan. Manfaatkan momentum ini untuk menanamkan "
                    f"kesadaran pencegahan. Fokus pada konten yang membahas cara mencegah "
                    f"kemunculan '{kw1}' dan '{kw2}' agar kondisi tidak kambuh."
                ),
                "content_pillars": (
                    f"1. Gaya hidup pencegahan {kw1}\n"
                    f"2. Pola makan anti-{topic}\n"
                    f"3. Checklist harian kesehatan {topic}\n"
                    f"4. Stres dan hubungannya dengan {kw2}"
                ),
                "call_to_action": (
                    f"Publikasikan konten pencegahan secara series (3-5 bagian) "
                    f"untuk membangun kebiasaan sehat audiens secara berkelanjutan."
                ),
            }
        else:
            return {
                "content_type": "Edukasi Solutif (Reguler)",
                "recommendation": (
                    f"Keluhan negatif tentang '{kw1}' dan '{kw2}' dalam konteks '{topic}' "
                    f"bersifat konsisten dan stabil. Ini menandakan kebutuhan edukasi yang "
                    f"berkelanjutan. Buat seri konten mingguan yang membahas solusi "
                    f"penanganan {topic} secara komprehensif."
                ),
                "content_pillars": (
                    f"1. Penjelasan medis penyebab {kw1}\n"
                    f"2. Pilihan pengobatan untuk {kw2}\n"
                    f"3. Perubahan gaya hidup yang terbukti efektif\n"
                    f"4. Q&A dengan pakar kesehatan"
                ),
                "call_to_action": (
                    f"Jadwalkan konten mingguan tentang {topic}. "
                    f"Gunakan format seri agar audiens kembali setiap minggu."
                ),
            }

    elif dominant_sentiment == "netral":
        if trend_status == "Naik":
            return {
                "content_type": "Edukasi Informatif (Momentum)",
                "recommendation": (
                    f"Perbincangan tentang '{topic}' sedang naik dengan nada netral. "
                    f"Audiens sedang aktif mencari informasi tentang '{kw1}' dan '{kw2}'. "
                    f"Ini adalah kesempatan emas untuk menjadi sumber informasi otoritatif."
                ),
                "content_pillars": (
                    f"1. Panduan lengkap memahami {topic} dari A-Z\n"
                    f"2. Apa itu {kw1}: penjelasan sederhana untuk awam\n"
                    f"3. Perbedaan {kw1} dan {kw2}: mana yang lebih perlu diperhatikan?\n"
                    f"4. Rekomendasi sumber informasi {topic} yang terpercaya"
                ),
                "call_to_action": (
                    f"Manfaatkan momentum perbincangan yang sedang naik. "
                    f"Publikasikan konten informatif panjang sekarang."
                ),
            }
        elif trend_status == "Turun":
            return {
                "content_type": "Pengingat Informatif",
                "recommendation": (
                    f"Perbincangan netral tentang '{topic}' sedang menurun. "
                    f"Waktu yang tepat untuk membuat konten pengingat yang membangkitkan "
                    f"kembali ketertarikan audiens pada topik '{kw1}' dan '{kw2}'."
                ),
                "content_pillars": (
                    f"1. Fakta mengejutkan tentang {kw1} yang jarang diketahui\n"
                    f"2. Mitos populer seputar {topic} yang perlu diluruskan\n"
                    f"3. Update terbaru riset medis tentang {kw2}\n"
                    f"4. Kuis interaktif: seberapa paham kamu tentang {topic}?"
                ),
                "call_to_action": (
                    f"Gunakan format yang mendorong interaksi (poll, kuis) "
                    f"untuk menghidupkan kembali engagement audiens."
                ),
            }
        else:
            return {
                "content_type": "Edukasi Informatif (Mitos vs Fakta)",
                "recommendation": (
                    f"Sentimen netral yang stabil pada topik '{topic}' mengindikasikan "
                    f"banyak audiens yang mencari informasi faktual tentang '{kw1}' dan '{kw2}'. "
                    f"Produksi konten klarifikasi berformat Mitos vs Fakta dan luruskan "
                    f"kesalahpahaman umum yang beredar di media sosial."
                ),
                "content_pillars": (
                    f"1. Mitos vs Fakta: {kw1} dan pengaruhnya terhadap {topic}\n"
                    f"2. Apakah {kw2} berbahaya? Ini kata para ahli\n"
                    f"3. Literasi kesehatan {topic}: panduan membaca riset medis\n"
                    f"4. Tanya jawab bersama dokter spesialis"
                ),
                "call_to_action": (
                    f"Libatkan tenaga medis profesional sebagai narasumber untuk "
                    f"meningkatkan kredibilitas konten {topic} Anda."
                ),
            }

    else:
        return {
            "content_type": "Inspiratif / Testimony Review",
            "recommendation": (
                f"Sentimen positif dominan pada topik '{topic}' — terutama seputar '{kw1}' "
                f"dan '{kw2}'. Audiens sedang dalam mode berbagi pengalaman positif. "
                f"Manfaatkan momentum ini untuk mengkurasi konten inspiratif berupa "
                f"testimoni pemulihan dan ulasan metode kesehatan yang efektif."
            ),
            "content_pillars": (
                f"1. Kisah sukses pemulihan dari {topic}: perjalanan nyata\n"
                f"2. Review jujur: metode yang membantu mengatasi {kw1}\n"
                f"3. Before & After: perubahan gaya hidup yang mengalahkan {kw2}\n"
                f"4. Komunitas {topic}: bergabung dan berbagi semangat bersama"
            ),
            "call_to_action": (
                f"Undang audiens untuk berbagi kisah pemulihan mereka sendiri. "
                f"User-Generated Content tentang {topic} sangat powerful "
                f"untuk membangun komunitas dan kepercayaan organik."
            ),
        }


def generate_content_recommendation(keyword: str) -> Dict[str, Any]:
    normalized_keyword = keyword.strip().lower()
    logger.info("DSS Engine: memulai analisis untuk keyword='%s'", normalized_keyword)

    logger.info("Langkah 1/7: Mengambil data dari database...")
    raw_data = get_tweets_by_keyword(normalized_keyword, limit=300)

    if not raw_data:
        raise ValueError(
            f"Tidak ada data tweet ditemukan untuk keyword '{normalized_keyword}'. "
            "Pastikan keyword valid dan data sudah tersedia di database."
        )

    logger.info("Berhasil mengambil %d record tweet.", len(raw_data))

    logger.info("Langkah 2/7: Menjalankan NLP preprocessing pipeline...")
    clean_texts = []
    dates = []

    for record in raw_data:
        source_text = record.get("normalisasi") or record.get("full_text") or ""
        cleaned = clean_text(source_text)
        clean_texts.append(cleaned)
        dates.append(record.get("created_at"))

    logger.info("Preprocessing selesai untuk %d teks.", len(clean_texts))

    logger.info("Langkah 3/7: Menjalankan inferensi sentimen IndoBERTweet...")
    sentiment_labels = predict_sentiment_batch(clean_texts)
    logger.info("Inferensi sentimen selesai.")

    logger.info("Langkah 4/7: Menghitung distribusi sentimen...")
    sentiment_distribution = _calculate_sentiment_distribution(sentiment_labels)
    dominant_sentiment = sentiment_distribution["dominant"]
    dominant_idx = sentiment_distribution["dominant_idx"]

    logger.info(
        "Distribusi: Negatif=%.1f%%, Netral=%.1f%%, Positif=%.1f%% | Dominan: %s",
        sentiment_distribution["percentages"]["negatif"],
        sentiment_distribution["percentages"]["netral"],
        sentiment_distribution["percentages"]["positif"],
        dominant_sentiment.upper(),
    )

    logger.info("Langkah 5/7: Menjalankan peramalan tren ARIMA...")
    valid_dates = [d for d in dates if d is not None]
    arima_result = forecast_trend(valid_dates)
    trend_status = arima_result["trend_status"]
    logger.info("Peramalan ARIMA selesai. Status tren: %s", trend_status)

    logger.info("Langkah 6/7: Mengekstrak kata kunci dari sentimen dominan...")
    top_keywords = _extract_top_keywords(
        clean_texts=clean_texts,
        dominant_label_idx=dominant_idx,
        all_labels=sentiment_labels,
        top_n=2,
    )

    logger.info(
        "Langkah 7/7: Mengevaluasi rule-based matrix (sentimen=%s, tren=%s)...",
        dominant_sentiment,
        trend_status,
    )
    content_recommendation = _generate_recommendation_text(
        dominant_sentiment=dominant_sentiment,
        trend_status=trend_status,
        top_keywords=top_keywords,
        keyword=normalized_keyword,
    )

    logger.info(
        "DSS Engine selesai. Tipe konten: '%s'",
        content_recommendation["content_type"],
    )

    return {
        "keyword":                normalized_keyword,
        "total_data":             len(raw_data),
        "sentiment_distribution": sentiment_distribution,
        "dominant_sentiment":     dominant_sentiment,
        "arima_forecast":         arima_result,
        "top_keywords":           top_keywords,
        "content_recommendation": content_recommendation,
    }
