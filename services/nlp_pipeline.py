# services/nlp_pipeline.py
# Modul pipeline pemrosesan teks (Natural Language Processing).
# Melakukan case folding, pembersihan noise, normalisasi slang, dan stopword removal
# khusus untuk teks berbahasa Indonesia dari Twitter.

import re
import logging

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from utils.slang_dict import SLANG_MAP

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Inisialisasi StopWord Remover Sastrawi.
# Instance dibuat sekali di level modul (module-level singleton) agar
# tidak ada overhead inisialisasi berulang setiap kali clean_text() dipanggil.
# -------------------------------------------------------------------

_stop_word_factory = StopWordRemoverFactory()
_stopword_remover = _stop_word_factory.create_stop_word_remover()

# Daftar stopword default Sastrawi untuk keperluan penghapusan per-kata
# (digunakan dalam proses tokenisasi manual jika dibutuhkan).
_stopword_set = set(_stop_word_factory.get_stop_words())


def _case_fold(text: str) -> str:
    """
    Melakukan case folding: mengubah seluruh karakter teks menjadi huruf kecil.

    Args:
        text (str): Teks input dalam format apapun.

    Returns:
        str: Teks yang seluruh karakternya sudah menjadi huruf kecil.
    """
    return text.lower()


def _clean_noise(text: str) -> str:
    """
    Membersihkan noise/bising dari teks Twitter menggunakan regex.

    Jenis noise yang dihapus:
    - URL (http, https, www)
    - Mention pengguna (@username)
    - Hashtag (#topik)
    - Penanda retweet (RT)
    - Angka (0-9)
    - Semua karakter non-alfabet dan non-spasi (tanda baca, emoji, dll.)
    - Spasi berlebih lebih dari satu (diratakan menjadi satu spasi)

    Args:
        text (str): Teks hasil case folding.

    Returns:
        str: Teks bersih tanpa noise.
    """
    # Hapus URL (mulai dengan http, https, atau www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Hapus mention (@username)
    text = re.sub(r"@\w+", " ", text)

    # Hapus hashtag (#topik)
    text = re.sub(r"#\w+", " ", text)

    # Hapus awalan retweet "rt" di awal teks (setelah case folding)
    text = re.sub(r"\brt\b", " ", text)

    # Hapus semua angka
    text = re.sub(r"\d+", " ", text)

    # Hapus semua karakter yang bukan huruf latin dan bukan spasi
    # (mencakup tanda baca, emoji, karakter khusus, dll.)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalisasi spasi berlebih (hapus spasi ganda atau lebih menjadi satu spasi)
    text = re.sub(r"\s+", " ", text)

    # Hapus spasi di awal dan akhir string
    text = text.strip()

    return text


def _normalize_slang(text: str) -> str:
    """
    Mengganti kata-kata slang/tidak baku dengan padanan kata baku
    menggunakan kamus SLANG_MAP dari utils/slang_dict.py.

    Pencocokan dilakukan per-kata (word boundary \\b) untuk menghindari
    penggantian kata yang hanya merupakan bagian dari kata lain.

    Args:
        text (str): Teks yang sudah dibersihkan dari noise.

    Returns:
        str: Teks dengan kata slang sudah diganti ke kata baku.
    """
    tokens = text.split()
    normalized_tokens = []

    for token in tokens:
        # Cari token di kamus SLANG_MAP.
        # Jika ditemukan, ganti dengan kata bakunya; jika tidak, pertahankan token asli.
        normalized_token = SLANG_MAP.get(token, token)
        normalized_tokens.append(normalized_token)

    return " ".join(normalized_tokens)


def _remove_stopwords(text: str) -> str:
    """
    Menghapus kata-kata stopword Bahasa Indonesia dari teks menggunakan Sastrawi.

    Sastrawi menggunakan daftar stopword bawaan yang komprehensif untuk Bahasa Indonesia.

    Args:
        text (str): Teks yang sudah dinormalisasi.

    Returns:
        str: Teks tanpa stopword.
    """
    return _stopword_remover.remove(text)


def clean_text(text: str) -> str:
    """
    Fungsi utama pipeline NLP. Menjalankan seluruh tahapan preprocessing
    teks secara berurutan:
    1. Case folding (lowercase)
    2. Pembersihan noise (URL, mention, hashtag, RT, angka, non-alfabet)
    3. Normalisasi kata slang (menggunakan SLANG_MAP)
    4. Penghapusan stopword (menggunakan Sastrawi)

    Args:
        text (str): Teks mentah tweet dari database (full_text atau normalisasi).
                    Jika input bukan string atau None/NaN, akan dikembalikan
                    string kosong untuk mencegah error downstream.

    Returns:
        str: Teks bersih siap digunakan sebagai input model IndoBERTweet.
    """
    # Guard clause: pastikan input adalah string yang valid
    if not isinstance(text, str) or not text.strip():
        return ""

    # Tahap 1: Case Folding
    text = _case_fold(text)

    # Tahap 2: Pembersihan Noise
    text = _clean_noise(text)

    # Tahap 3: Normalisasi Kata Slang
    text = _normalize_slang(text)

    # Tahap 4: Penghapusan Stopword
    text = _remove_stopwords(text)

    # Normalisasi akhir: hapus spasi berlebih yang mungkin muncul setelah stopword removal
    text = re.sub(r"\s+", " ", text).strip()

    return text
