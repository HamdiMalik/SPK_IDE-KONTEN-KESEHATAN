# services/arima_service.py
# Modul layanan peramalan tren volume perbincangan menggunakan model ARIMA.
# Menerima list timestamp dari database, melakukan agregasi harian,
# dan meramal 3 hari ke depan untuk menentukan tren konten.

import logging
from typing import List, Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Konstanta konfigurasi ARIMA
# Order (p, d, q) = (1, 1, 1) adalah parameter default yang umum digunakan
# sebagai baseline untuk deret waktu non-stasioner dengan komponen autoregresif.
# p=1: menggunakan 1 lag autoregresif
# d=1: diferensiasi 1 kali untuk membuat deret stasioner
# q=1: menggunakan 1 lag moving average
# -------------------------------------------------------------------

_ARIMA_ORDER = (1, 1, 1)
_FORECAST_STEPS = 3


def forecast_trend(dates: List[Any]) -> Dict[str, Any]:
    """
    Melakukan peramalan tren volume perbincangan tweet berdasarkan data historis tanggal.

    Alur pemrosesan:
    1. Konversi list datetime ke pandas Series.
    2. Agregasi berdasarkan tanggal (hitung jumlah tweet per hari).
    3. Isi tanggal yang kosong (missing dates) dengan nilai 0 menggunakan reindex.
    4. Fit model ARIMA dengan order (1,1,1) ke data historis.
    5. Ramal volume 3 hari ke depan.
    6. Tentukan status tren: "Naik", "Turun", atau "Stabil".

    Args:
        dates (List[Any]): List objek datetime atau string timestamp yang merepresentasikan
                           kolom 'created_at' dari setiap tweet hasil query database.

    Returns:
        Dict[str, Any]: Dictionary berisi:
            - 'historical_dates'  (List[str])   : List tanggal historis (format YYYY-MM-DD).
            - 'historical_volume' (List[int])   : Jumlah tweet per tanggal historis.
            - 'forecast_dates'    (List[str])   : List 3 tanggal hasil prediksi (format YYYY-MM-DD).
            - 'forecast_volume'   (List[float]) : Estimasi volume tweet untuk 3 hari ke depan.
            - 'trend_status'      (str)         : Status tren: "Naik", "Turun", atau "Stabil".
            - 'trend_description' (str)         : Deskripsi teks status tren untuk UI.

    Raises:
        ValueError: Jika list dates kosong atau tidak cukup data untuk fitting ARIMA.
        RuntimeError: Jika terjadi error selama proses fitting atau forecasting ARIMA.
    """
    if not dates:
        raise ValueError(
            "List tanggal (dates) tidak boleh kosong untuk melakukan peramalan ARIMA."
        )

    logger.info(
        "Memulai peramalan ARIMA untuk %d data timestamp...", len(dates)
    )

    # -------------------------------------------------------------------
    # Tahap 1: Konversi dan Agregasi Data Deret Waktu
    # -------------------------------------------------------------------

    # Konversi list tanggal ke pandas Series dengan tipe datetime
    date_series = pd.to_datetime(pd.Series(dates), errors="coerce")

    # Hapus nilai NaT (Not a Time) yang mungkin muncul dari konversi gagal
    date_series = date_series.dropna()

    if date_series.empty:
        raise ValueError(
            "Tidak ada data tanggal valid setelah konversi. "
            "Pastikan kolom 'created_at' memiliki format datetime yang benar."
        )

    # Ekstrak hanya bagian tanggal (tanpa jam/menit/detik) untuk agregasi harian
    date_only_series = date_series.dt.date

    # Hitung jumlah tweet per tanggal menggunakan groupby + size()
    daily_counts = date_only_series.groupby(date_only_series).size()

    # Konversi index ke DatetimeIndex agar bisa digunakan reindex dengan frekuensi harian
    daily_counts.index = pd.to_datetime(daily_counts.index)
    daily_counts = daily_counts.sort_index()

    # -------------------------------------------------------------------
    # Tahap 2: Isi Missing Dates dengan Nilai 0 (Time Series Continuity)
    # Ini memastikan deret waktu kontinu tanpa jeda, yang merupakan
    # prasyarat penting untuk model ARIMA.
    # -------------------------------------------------------------------

    full_date_range = pd.date_range(
        start=daily_counts.index.min(),
        end=daily_counts.index.max(),
        freq="D",  # Frekuensi harian (daily)
    )

    # Reindex: tanggal yang tidak ada dalam data asli akan diisi NaN,
    # kemudian kita ganti NaN dengan 0.
    daily_counts_complete = daily_counts.reindex(full_date_range, fill_value=0)

    logger.info(
        "Data historis: %d hari (dari %s hingga %s). Total tweet: %d.",
        len(daily_counts_complete),
        daily_counts_complete.index[0].strftime("%Y-%m-%d"),
        daily_counts_complete.index[-1].strftime("%Y-%m-%d"),
        int(daily_counts_complete.sum()),
    )

    # -------------------------------------------------------------------
    # Tahap 3: Fitting Model ARIMA
    # Jika data historis kurang dari 3 titik, ARIMA tidak bisa di-fit.
    # Kita tangani kasus edge ini dengan memberikan fallback sederhana.
    # -------------------------------------------------------------------

    if len(daily_counts_complete) < 3:
        logger.warning(
            "Data historis terlalu sedikit (%d hari) untuk fitting ARIMA. "
            "Menggunakan fallback nilai rata-rata.",
            len(daily_counts_complete),
        )
        avg_volume = float(daily_counts_complete.mean())
        last_historical_date = daily_counts_complete.index[-1]
        forecast_dates = [
            (last_historical_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(_FORECAST_STEPS)
        ]
        forecast_values = [round(avg_volume, 2)] * _FORECAST_STEPS
        trend_status = "Stabil"
        trend_description = "Data historis terlalu sedikit, tren diestimasi stabil."

        return {
            "historical_dates":  [d.strftime("%Y-%m-%d") for d in daily_counts_complete.index],
            "historical_volume": daily_counts_complete.tolist(),
            "forecast_dates":    forecast_dates,
            "forecast_volume":   forecast_values,
            "trend_status":      trend_status,
            "trend_description": trend_description,
        }

    try:
        # Fit model ARIMA ke data historis lengkap
        arima_model = ARIMA(
            daily_counts_complete.values.astype(float),
            order=_ARIMA_ORDER,
        )
        arima_result = arima_model.fit()

        logger.info(
            "Model ARIMA(%d,%d,%d) berhasil di-fit. AIC: %.4f",
            *_ARIMA_ORDER,
            arima_result.aic,
        )

        # -------------------------------------------------------------------
        # Tahap 4: Peramalan (Forecasting) 3 Langkah ke Depan
        # -------------------------------------------------------------------

        forecast_output = arima_result.forecast(steps=_FORECAST_STEPS)

        # Pastikan nilai forecast tidak negatif (volume tidak mungkin negatif)
        forecast_values = [max(0.0, round(float(val), 2)) for val in forecast_output]

        # Generate tanggal untuk setiap langkah prediksi
        last_historical_date = daily_counts_complete.index[-1]
        forecast_dates = [
            (last_historical_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(_FORECAST_STEPS)
        ]

        # -------------------------------------------------------------------
        # Tahap 5: Penentuan Status Tren
        # Bandingkan nilai akhir prediksi dengan nilai akhir historis.
        # Toleransi 10% digunakan untuk menentukan kategori "Stabil"
        # agar tidak terlalu sensitif terhadap fluktuasi kecil.
        # -------------------------------------------------------------------

        last_historical_value = float(daily_counts_complete.iloc[-1])
        last_forecast_value = forecast_values[-1]

        # Hitung persentase perubahan relatif
        if last_historical_value > 0:
            change_percentage = (
                (last_forecast_value - last_historical_value) / last_historical_value
            ) * 100
        else:
            # Jika nilai historis terakhir adalah 0, gunakan perbandingan absolut
            change_percentage = last_forecast_value * 100

        _STABILITY_THRESHOLD = 10.0  # Persentase toleransi untuk kategori "Stabil"

        if change_percentage > _STABILITY_THRESHOLD:
            trend_status = "Naik"
            trend_description = (
                f"Volume perbincangan diprediksi NAIK sekitar "
                f"{abs(change_percentage):.1f}% dalam 3 hari ke depan."
            )
        elif change_percentage < -_STABILITY_THRESHOLD:
            trend_status = "Turun"
            trend_description = (
                f"Volume perbincangan diprediksi TURUN sekitar "
                f"{abs(change_percentage):.1f}% dalam 3 hari ke depan."
            )
        else:
            trend_status = "Stabil"
            trend_description = (
                "Volume perbincangan diprediksi STABIL "
                f"(perubahan ±{abs(change_percentage):.1f}%) dalam 3 hari ke depan."
            )

        logger.info(
            "Peramalan ARIMA selesai. Tren: %s | Forecast: %s",
            trend_status,
            forecast_values,
        )

        return {
            "historical_dates":  [d.strftime("%Y-%m-%d") for d in daily_counts_complete.index],
            "historical_volume": [int(v) for v in daily_counts_complete.tolist()],
            "forecast_dates":    forecast_dates,
            "forecast_volume":   forecast_values,
            "trend_status":      trend_status,
            "trend_description": trend_description,
        }

    except Exception as exc:
        logger.error(
            "Terjadi error saat fitting atau forecasting ARIMA: %s",
            str(exc),
            exc_info=True,
        )
        raise RuntimeError(
            f"Gagal melakukan peramalan ARIMA: {str(exc)}"
        ) from exc
