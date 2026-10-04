"""Collector WikiSpike: menambahkan hitungan suntingan per menit ke data/edit_counts.csv.

Dijalankan berkala (GitHub Actions tiap 5 menit). Aman terhadap jadwal yang
terlewat: ia melanjutkan dari menit terakhir yang tersimpan. Menit yang tidak
bisa diambil lengkap dibiarkan kosong (bukan nol), supaya baseline tidak bias.
"""
import os

import pandas as pd

from wikispike_core import HISTORY_PATH, aggregate_minutes, fetch_changes, safe_end_minute

KEEP_DAYS = 60            # riwayat lebih tua dipangkas agar file tidak membengkak
INITIAL_BACKFILL_MIN = 180


def main():
    end = safe_end_minute()
    old = None
    if os.path.exists(HISTORY_PATH):
        old = pd.read_csv(HISTORY_PATH, parse_dates=["minute"], index_col="minute")
        old.index = pd.to_datetime(old.index, utc=True)
        old["top_page"] = old["top_page"].fillna("")
        start = old.index.max() + pd.Timedelta(minutes=1)
    else:
        start = end - pd.Timedelta(minutes=INITIAL_BACKFILL_MIN)

    if start >= end:
        print("Tidak ada menit baru.")
        return

    raw, first_full = fetch_changes(start)
    if first_full is None or first_full >= end:
        print("Data tidak lengkap/terlalu baru, dilewati.")
        return

    new = aggregate_minutes(raw, first_full, end)
    if len(new) >= 3 and new["edit_count"].sum() == 0:
        print("Hasil semua nol, dicurigai kegagalan API; tidak disimpan.")
        return

    merged = new if old is None else pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged = merged[merged.index >= end - pd.Timedelta(days=KEEP_DAYS)]

    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    merged.to_csv(HISTORY_PATH, index_label="minute")
    print(f"Ditambahkan {len(new)} menit; total {len(merged)} baris; "
          f"terakhir {merged.index.max()}.")


if __name__ == "__main__":
    main()
