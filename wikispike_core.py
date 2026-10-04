"""Akses Wikimedia (Action API + EventStreams) dan agregasi per menit.
Dipakai bersama oleh collector.py dan app.py."""
import json
import time

import pandas as pd
import requests

STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
HEADERS = {
    # Wajib: identitas kustom agar tidak kena 403. Ganti dengan kontak Anda.
    "User-Agent": "WikiSpike/4.0 (https://github.com/REY-gifthub/wiki-streamlit; muhammadyuzaulauladi@gmail.com)"
}
DEFAULT_WIKIS = ["enwiki", "idwiki", "jawiki"]
COLUMNS = ["edit_count", "bot_count", "unique_editors", "top_page", "top_page_share"]


def history_path(wiki: str) -> str:
    """enwiki tetap memakai nama berkas lama agar data v3 tidak hilang."""
    return "data/edit_counts.csv" if wiki == "enwiki" else f"data/edit_counts_{wiki}.csv"


def wiki_host(wiki: str) -> str:
    """'idwiki' -> 'id.wikipedia.org' (hanya untuk proyek Wikipedia)."""
    if not wiki.endswith("wiki") or len(wiki) <= 4:
        raise ValueError(f"ID wiki tidak dikenali: {wiki}")
    return f"{wiki[:-4]}.wikipedia.org"


def safe_end_minute(now=None) -> pd.Timestamp:
    """Batas atas (eksklusif) menit yang sudah lengkap.
    Mundur 30 detik untuk memberi ruang bagi tunda replikasi data."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return (now - pd.Timedelta(seconds=30)).floor("min")


# ---------------------------------------------------------------- Action API (live, satu wiki)
def fetch_changes(start: pd.Timestamp, wiki: str = "enwiki", max_pages: int = 60,
                  pause: float = 0.1, session=None):
    """Tarik suntingan dari sekarang mundur sampai `start` (Timestamp UTC).

    Return (raw_df, first_full_minute). `first_full_minute` adalah menit paling awal
    yang datanya terjamin lengkap: jika paginasi terpotong `max_pages`, menit-menit
    lebih lama TIDAK boleh dianggap nol (hanya tidak teramati).
    """
    s = session or requests.Session()
    s.headers.update(HEADERS)
    api = f"https://{wiki_host(wiki)}/w/api.php"
    params = {
        "action": "query", "list": "recentchanges",
        "rcprop": "title|timestamp|user|flags|ids|sizes",
        "rctype": "edit|new", "rclimit": 500, "rcdir": "older",
        "rcend": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "json", "formatversion": 2,
    }
    rows, complete = [], False
    for _ in range(max_pages):
        r = s.get(api, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        rows.extend(data.get("query", {}).get("recentchanges", []))
        if "continue" not in data:
            complete = True
            break
        params.update(data["continue"])
        time.sleep(pause)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, (start.ceil("min") if complete else None)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["bot"] = df["bot"].fillna(False).astype(bool) if "bot" in df else False
    df["user"] = df["user"].fillna("") if "user" in df else ""
    if "rcid" in df:
        df = df.drop_duplicates(subset="rcid")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if complete:
        first_full = start.ceil("min")
    else:  # terpotong: menit tertua yang terambil masih parsial
        first_full = df["timestamp"].iloc[0].floor("min") + pd.Timedelta(minutes=1)
    return df, first_full


# ---------------------------------------------------------------- EventStreams (collector, multi-wiki)
def stream_changes(since: pd.Timestamp, until: pd.Timestamp, wikis, max_seconds: float = 240,
                   url: str = STREAM_URL):
    """Konsumsi EventStreams `recentchange` dari `since` (replay historis) sampai event
    berstempel >= `until`. Satu koneksi mencakup semua wiki; disaring di sisi klien.

    Return (df, covered_until). Menit < covered_until dijamin lengkap; bila koneksi
    putus atau waktu habis, covered_until = menit terakhir yang aman, sisanya dilanjutkan
    pada run berikutnya. df berkolom: timestamp, wiki, title, user, bot.
    """
    wikis = set(wikis)
    rows, last_ts, reached = [], None, False
    deadline = time.monotonic() + max_seconds
    headers = {**HEADERS, "Accept": "text/event-stream"}
    params = {"since": since.strftime("%Y-%m-%dT%H:%M:%SZ")}
    stop_at = int((until + pd.Timedelta(seconds=10)).timestamp())  # toleransi urutan

    try:
        with requests.get(url, params=params, headers=headers, stream=True,
                          timeout=(10, 45)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if time.monotonic() > deadline:
                    break
                if not line or not line.startswith(b"data:"):
                    continue
                try:
                    ev = json.loads(line[5:])
                except ValueError:
                    continue
                ts = ev.get("timestamp")
                if not isinstance(ts, (int, float)):
                    continue
                last_ts = ts if last_ts is None else max(last_ts, ts)
                if ts >= stop_at:
                    reached = True
                    break
                if ev.get("type") in ("edit", "new") and ev.get("wiki") in wikis:
                    rows.append((ts, ev["wiki"], ev.get("title", ""),
                                 ev.get("user", ""), bool(ev.get("bot", False))))
    except (requests.RequestException, OSError) as e:
        print(f"Koneksi stream terputus: {e}")

    if last_ts is None:
        return pd.DataFrame(columns=["timestamp", "wiki", "title", "user", "bot"]), None
    df = pd.DataFrame(rows, columns=["ts", "wiki", "title", "user", "bot"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop(columns="ts").sort_values("timestamp").reset_index(drop=True)
    if reached:
        covered = until
    else:
        covered = (pd.Timestamp(last_ts, unit="s", tz="UTC") - pd.Timedelta(seconds=10)).floor("min")
    return df, covered


# ---------------------------------------------------------------- agregasi
def aggregate_minutes(raw: pd.DataFrame, start_minute, end_minute) -> pd.DataFrame:
    """Agregasi per menit pada [start_minute, end_minute). Menit tanpa suntingan = 0."""
    full = pd.date_range(start_minute, end_minute - pd.Timedelta(minutes=1),
                         freq="1min", name="minute")
    if len(full) == 0:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.DataFrame(index=full)

    r = raw
    if not raw.empty:
        r = raw[(raw["timestamp"] >= start_minute) & (raw["timestamp"] < end_minute)].copy()
    if not r.empty:
        r["minute"] = r["timestamp"].dt.floor("min")
        g = r.groupby("minute")
        agg = pd.DataFrame({
            "edit_count": g.size(),
            "bot_count": g["bot"].sum().astype(int),
            "unique_editors": g["user"].nunique(),
        })
        pt = r.groupby(["minute", "title"]).size().rename("n").reset_index()
        top = pt.loc[pt.groupby("minute")["n"].idxmax()].set_index("minute")
        agg["top_page"] = top["title"]
        agg["top_page_share"] = (top["n"] / agg["edit_count"]).round(3)
        out = out.join(agg)
    for col, fill in [("edit_count", 0), ("bot_count", 0), ("unique_editors", 0),
                      ("top_page", ""), ("top_page_share", 0.0)]:
        if col not in out:
            out[col] = fill
        out[col] = out[col].fillna(fill)
    for col in ["edit_count", "bot_count", "unique_editors"]:
        out[col] = out[col].astype(int)
    return out[COLUMNS]