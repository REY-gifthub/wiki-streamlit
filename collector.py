"""Collector WikiSpike v4 (multi-wiki, EventStreams).

Menambahkan hitungan suntingan per menit ke data/edit_counts*.csv untuk setiap wiki.
Satu koneksi EventStreams (replay lewat parameter `since`) mencakup semua wiki.
Aman terhadap jadwal yang terlewat: tiap wiki dilanjutkan dari menit terakhirnya.
Menit yang tidak bisa diambil lengkap dibiarkan kosong (bukan nol).

Atur daftar wiki lewat env WIKISPIKE_WIKIS, mis. "enwiki,idwiki,jawiki".
"""
import os

import pandas as pd

from wikispike_core import (DEFAULT_WIKIS, aggregate_minutes, history_path,
                            safe_end_minute, stream_changes)

KEEP_DAYS = 60
INITIAL_BACKFILL_MIN = 60   # wiki baru: ambil 60 menit terakhir (hemat bandwidth)
MAX_REPLAY_DAYS = 6         # retensi EventStreams sekitar 7-31 hari; pakai batas aman


def load_old(wiki):
    path = history_path(wiki)
    if not os.path.exists(path):
        return None
    old = pd.read_csv(path, parse_dates=["minute"], index_col="minute")
    old.index = pd.to_datetime(old.index, utc=True)
    old["top_page"] = old["top_page"].fillna("")
    return old


def main():
    wikis = [w.strip() for w in os.environ.get("WIKISPIKE_WIKIS", ",".join(DEFAULT_WIKIS)).split(",") if w.strip()]
    end = safe_end_minute()
    floor_start = end - pd.Timedelta(days=MAX_REPLAY_DAYS)

    olds, starts = {}, {}
    for w in wikis:
        olds[w] = load_old(w)
        s = (olds[w].index.max() + pd.Timedelta(minutes=1)) if olds[w] is not None \
            else end - pd.Timedelta(minutes=INITIAL_BACKFILL_MIN)
        starts[w] = max(s, floor_start)

    pending = {w: s for w, s in starts.items() if s < end}
    if not pending:
        print("Tidak ada menit baru.")
        return

    since = min(pending.values()) - pd.Timedelta(seconds=30)  # margin
    raw, covered_until = stream_changes(since, end, list(pending))
    if covered_until is None or raw.empty:
        print("Tidak ada event yang diterima; dilewati.")
        return
    stop = min(end, covered_until)
    print(f"{len(raw)} event diterima; lengkap sampai {stop}.")

    os.makedirs("data", exist_ok=True)
    for w, start in pending.items():
        if start >= stop:
            continue
        new = aggregate_minutes(raw[raw["wiki"] == w], start, stop)
        old = olds[w]
        merged = new if old is None else pd.concat([old, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged = merged[merged.index >= end - pd.Timedelta(days=KEEP_DAYS)]
        merged.to_csv(history_path(w), index_label="minute")
        print(f"[{w}] +{len(new)} menit; total {len(merged)}; terakhir {merged.index.max()}.")


if __name__ == "__main__":
    main()