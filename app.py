"""
WikiSpike v4 - Deteksi Anomali Aktivitas Suntingan Wikipedia (multi-wiki)
Baseline musiman dari riwayat collector + shifted Robust Modified Z-Score.

Jalankan:  streamlit run app.py
Opsional:  st.secrets["DATA_BASE_URL"] = folder raw data, mis.
           https://raw.githubusercontent.com/USER/REPO/main/data
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from wikispike_core import (DEFAULT_WIKIS, aggregate_minutes, fetch_changes,
                            history_path, safe_end_minute)
from wikispike_stats import detect, run_detection, simulate_spikes

st.set_page_config(page_title="WikiSpike", layout="wide")
st.title("WikiSpike - Deteksi Anomali Suntingan Wikipedia")


# ---------------------------------------------------------------- data
@st.cache_data(ttl=45, show_spinner=False)
def load_live(wiki: str, lookback: int) -> pd.DataFrame:
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=lookback)
    raw, first_full = fetch_changes(start, wiki=wiki)
    if first_full is None:
        return pd.DataFrame()
    return aggregate_minutes(raw, first_full, safe_end_minute())


@st.cache_data(ttl=60, show_spinner=False)
def load_history(wiki: str) -> pd.DataFrame:
    path = history_path(wiki)
    try:
        base = st.secrets["DATA_BASE_URL"].rstrip("/")
        src = f"{base}/{os.path.basename(path)}"
    except Exception:
        src = path
    try:
        df = pd.read_csv(src, parse_dates=["minute"], index_col="minute")
    except Exception:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index, utc=True)
    df["top_page"] = df["top_page"].fillna("")
    return df


def merge(hist: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return live
    if live.empty:
        return hist
    return pd.concat([hist, live[~live.index.isin(hist.index)]]).sort_index()


def to_counts(merged: pd.DataFrame, exclude_bots: bool):
    full = merged.reindex(pd.date_range(merged.index.min(), merged.index.max(), freq="1min"))
    counts = full["edit_count"].astype(float)
    if exclude_bots:
        counts = counts - full["bot_count"].astype(float)
    return full, counts


def days_of(hist: pd.DataFrame) -> float:
    return (hist.index.max() - hist.index.min()).total_seconds() / 86400 if not hist.empty else 0.0


def hint(row) -> str:
    """Petunjuk heuristik (bukan classifier) untuk membantu menafsirkan anomali."""
    if row["bot_share"] >= 0.6:
        return "dominan bot"
    if row["top_page_share"] >= 0.25:
        return "terpusat pada satu halaman"
    return "tersebar"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Panel Kontrol")
    wiki = st.selectbox("Wiki", DEFAULT_WIKIS)
    mode = st.radio("Mode baseline", ["Musiman (riwayat collector)", "Lokal (live saja)"])
    window = st.slider("Window Size residual/lokal (menit)", 3, 20, 5)
    z_thr = st.slider("Z-Threshold", 2.0, 10.0, 3.0, 0.5)
    one_sided = st.checkbox("Uji satu sisi (lonjakan saja)", True)
    m = st.slider("Persistensi: jendela m (menit)", 1, 5, 1)
    k = int(st.number_input("Persistensi: minimal k menit melewati ambang", 1, m, 1))
    exclude_bots = st.checkbox("Kecualikan suntingan bot", False)
    lookback = st.slider("Rentang data live (menit)", 30, 180, 60, 10)
    seasonal_pref = mode.startswith("Musiman")
    days, half_w, same_dt, view_hours = 7, 10, True, 24
    if seasonal_pref:
        st.subheader("Baseline musiman")
        days = st.slider("Jumlah hari referensi", 2, 28, 7)
        half_w = st.slider("Lebar lingkungan +- menit", 0, 30, 10)
        same_dt = st.checkbox("Bandingkan hari kerja dengan hari kerja", True,
                              help="Akhir pekan hanya dibandingkan dengan akhir pekan.")
        view_hours = st.slider("Rentang tampilan (jam)", 3, 72, 24)
    auto = st.checkbox("Auto-refresh tiap 60 detik", False)
    if st.button("Refresh Data"):
        st.cache_data.clear()

det_kw = dict(window=window, z_thr=z_thr, one_sided=one_sided, k=k, m=m)
seasonal_cfg = dict(days=days, half_width=half_w, same_daytype=same_dt)


# ---------------------------------------------------------------- tampilan utama
@st.fragment(run_every=60 if auto else None)
def dashboard():
    try:
        live = load_live(wiki, lookback)
    except (requests.exceptions.RequestException, RuntimeError, ValueError) as e:
        live = pd.DataFrame()
        st.warning(f"Gagal menarik data live dari Wikimedia API: {e}")

    hist = load_history(wiki)
    hist_days = days_of(hist)
    seasonal = seasonal_pref
    if seasonal and hist_days < 2:
        st.warning("Riwayat collector belum cukup (butuh minimal 2 hari). "
                   "Beralih ke mode lokal sementara.")
        seasonal = False

    merged = merge(hist, live) if seasonal else live
    if merged.empty:
        st.error("Tidak ada data. Periksa koneksi atau jalankan collector.")
        return
    full, counts_all = to_counts(merged, exclude_bots)
    if len(counts_all.dropna()) <= window:
        st.warning(f"Data baru {len(counts_all.dropna())} menit, kurang dari Window Size ({window}).")
        return
    if seasonal and hist_days < days:
        st.info(f"Riwayat baru {hist_days:.1f} hari (< {days} hari referensi); "
                "baseline musiman memakai hari yang tersedia.")

    cview, S, result_all, warm = run_detection(
        counts_all, view_hours * 60 if seasonal else 0, det_kw, seasonal_cfg if seasonal else None)
    result = result_all.iloc[warm:] if len(result_all) > warm else result_all
    anom = result[result["is_anomaly"]]
    lag = (pd.Timestamp.now(tz="UTC") - full.index.max()).total_seconds() / 60

    tab_det, tab_multi, tab_eval = st.tabs(["Deteksi", "Ringkasan wiki", "Evaluasi (simulasi)"])

    with tab_det:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mode", "Musiman" if seasonal else "Lokal")
        c2.metric("Riwayat tersimpan (hari)", f"{hist_days:.1f}")
        c3.metric("Alarm", len(anom))
        c4.metric("Data terakhir (menit lalu)", f"{lag:.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.index, y=result["edit_count"], mode="lines",
                                 name="Aktual", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=result.index, y=result["expected"], mode="lines",
                                 name="Baseline ekspektasi", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=anom.index, y=anom["edit_count"], mode="markers",
                                 name="Anomali", marker=dict(symbol="x", color="red", size=11)))
        fig.update_layout(xaxis_title="Waktu (UTC)", yaxis_title="Suntingan / menit",
                          hovermode="x unified", height=450)
        st.plotly_chart(fig)

        if not anom.empty:
            st.subheader("Konteks anomali")
            ctx = full.loc[anom.index, ["unique_editors", "top_page", "top_page_share", "bot_count"]].copy()
            ctx["bot_share"] = (ctx["bot_count"] / full.loc[anom.index, "edit_count"]).round(2)
            ctx["petunjuk"] = ctx.apply(hint, axis=1)
            ctx.insert(0, "edit_count", anom["edit_count"])
            ctx.insert(1, "mod_z", anom["mod_z"].round(2))
            st.dataframe(ctx.drop(columns="bot_count"))
            st.caption("Petunjuk bersifat heuristik (bot >= 60%, halaman teratas >= 25%), "
                       "bukan klasifikasi terlatih.")

        st.subheader("Tabel Kalkulasi")
        st.dataframe(result.round(3).rename(columns={
            "edit_count": "Edit Count", "expected": "Ekspektasi", "rolling_mad": "MAD residual",
            "mod_z": "Mod Z-Score", "flag_raw": "Lewat ambang", "is_anomaly": "Anomaly Status"}),
            )

    with tab_multi:
        st.write("Alarm 6 jam terakhir per wiki, dari **riwayat collector** saja (tanpa data live).")
        rows = []
        for w in DEFAULT_WIKIS:
            h = load_history(w)
            d = days_of(h)
            if h.empty or d < 2:
                rows.append({"Wiki": w, "Riwayat (hari)": round(d, 1), "Data terakhir (menit lalu)": None,
                             "Alarm 6 jam": None, "Catatan": "riwayat belum cukup"})
                continue
            f, c = to_counts(h, exclude_bots)
            _, _, res, wm = run_detection(c, 360, det_kw, seasonal_cfg)
            res = res.iloc[wm:]
            lg = (pd.Timestamp.now(tz="UTC") - f.index.max()).total_seconds() / 60
            rows.append({"Wiki": w, "Riwayat (hari)": round(d, 1),
                         "Data terakhir (menit lalu)": round(lg), "Alarm 6 jam": int(res["is_anomaly"].sum()),
                         "Catatan": ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    with tab_eval:
        st.write("Lonjakan sintetis disuntikkan ke data nyata untuk mengukur **recall**; "
                 "laju alarm pada data asli adalah perkiraan **batas atas** alarm palsu "
                 "(data asli mungkin memuat lonjakan sungguhan).")
        duration = st.slider("Durasi lonjakan sintetis (menit)", 1, 5, 2)
        mults = st.multiselect("Pengali lonjakan", [1.25, 1.5, 2.0, 3.0, 5.0], default=[1.5, 2.0, 3.0])
        if st.button("Jalankan simulasi") and mults:
            st.session_state["sim"] = simulate_spikes(cview, det_kw, mults, duration, seasonal=S)
            st.session_state["sim_ok"] = True
        if st.session_state.get("sim_ok"):
            out = st.session_state["sim"]
            if out is None:
                st.warning("Data valid terlalu pendek. Perbesar rentang tampilan/data atau kecilkan window.")
            else:
                tbl, fa, nvalid, nspk = out
                a, b = st.columns(2)
                a.metric("Menit beralarm per jam (data asli)", f"{fa:.2f}")
                b.metric("Menit valid / lonjakan disuntik", f"{nvalid} / {nspk}")
                st.dataframe(tbl.style.format({"Recall": "{:.0%}"}))


dashboard()