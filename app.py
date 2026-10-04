"""
WikiSpike v3 - Deteksi Anomali Aktivitas Suntingan Wikipedia
Baseline musiman dari riwayat collector + shifted Robust Modified Z-Score.

Jalankan:  streamlit run app.py
Opsional:  st.secrets["DATA_URL"] = URL raw CSV riwayat (mis. raw.githubusercontent.com/...)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from wikispike_core import HISTORY_PATH, aggregate_minutes, fetch_changes, safe_end_minute
from wikispike_stats import detect, seasonal_baseline, simulate_spikes

st.set_page_config(page_title="WikiSpike", layout="wide")
st.title("WikiSpike - Deteksi Anomali Suntingan Wikipedia")


# ---------------------------------------------------------------- data
@st.cache_data(ttl=60, show_spinner=False)
def load_live(lookback: int) -> pd.DataFrame:
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=lookback)
    raw, first_full = fetch_changes(start)
    if first_full is None:
        return pd.DataFrame()
    return aggregate_minutes(raw, first_full, safe_end_minute())


@st.cache_data(ttl=60, show_spinner=False)
def load_history() -> pd.DataFrame:
    try:
        src = st.secrets["DATA_URL"]
    except Exception:
        src = HISTORY_PATH
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
    mode = st.radio("Mode baseline", ["Musiman (riwayat collector)", "Lokal (live saja)"])
    window = st.slider("Window Size residual/lokal (menit)", 3, 20, 5)
    z_thr = st.slider("Z-Threshold", 2.0, 10.0, 3.0, 0.5)
    one_sided = st.checkbox("Uji satu sisi (lonjakan saja)", True)
    m = st.slider("Persistensi: jendela m (menit)", 1, 5, 1)
    k = int(st.number_input("Persistensi: minimal k menit melewati ambang", 1, m, 1))
    exclude_bots = st.checkbox("Kecualikan suntingan bot", False)
    lookback = st.slider("Rentang data live (menit)", 30, 180, 60, 10)
    if mode.startswith("Musiman"):
        st.subheader("Baseline musiman")
        days = st.slider("Jumlah hari referensi", 2, 28, 7)
        half_w = st.slider("Lebar lingkungan +- menit", 0, 30, 10)
        same_dt = st.checkbox("Bandingkan hari kerja dengan hari kerja", True,
                              help="Akhir pekan hanya dibandingkan dengan akhir pekan.")
        view_hours = st.slider("Rentang tampilan (jam)", 3, 72, 24)
    if st.button("Refresh Data"):
        st.cache_data.clear()

# ---------------------------------------------------------------- muat & gabung
try:
    live = load_live(lookback)
except (requests.exceptions.RequestException, RuntimeError) as e:
    live = pd.DataFrame()
    st.warning(f"Gagal menarik data live dari Wikimedia API: {e}")

hist = load_history()
hist_days = ((hist.index.max() - hist.index.min()).total_seconds() / 86400) if not hist.empty else 0.0

seasonal_mode = mode.startswith("Musiman")
if seasonal_mode and hist_days < 2:
    st.warning("Riwayat collector belum cukup (butuh minimal 2 hari). "
               "Beralih ke mode lokal sementara.")
    seasonal_mode = False

merged = merge(hist, live) if seasonal_mode else live
if merged.empty:
    st.error("Tidak ada data. Periksa koneksi atau jalankan collector.")
    st.stop()

full = merged.reindex(pd.date_range(merged.index.min(), merged.index.max(), freq="1min"))
counts_all = full["edit_count"].astype(float)
if exclude_bots:
    counts_all = counts_all - full["bot_count"].astype(float)

if len(counts_all.dropna()) <= window:
    st.warning(f"Data baru {len(counts_all.dropna())} menit, kurang dari Window Size ({window}).")
    st.stop()

det_kw = dict(window=window, z_thr=z_thr, one_sided=one_sided, k=k, m=m)
warm = window + m + 2

if seasonal_mode:
    tlen = min(len(full), view_hours * 60 + warm)
    tidx = full.index[-tlen:]
    S = seasonal_baseline(np.sqrt(counts_all.clip(lower=0)), tidx, days=days,
                          half_width=half_w, same_daytype=same_dt)
    cview = counts_all.loc[tidx]
    if hist_days < days:
        st.info(f"Riwayat baru {hist_days:.1f} hari (< {days} hari referensi); "
                "baseline musiman memakai hari yang tersedia.")
else:
    S, cview = None, counts_all

result_all = detect(cview, seasonal=S, **det_kw)
result = result_all.iloc[warm:] if len(result_all) > warm else result_all
anom = result[result["is_anomaly"]]

lag = (pd.Timestamp.now(tz="UTC") - full.index.max()).total_seconds() / 60
tab_det, tab_eval = st.tabs(["Deteksi", "Evaluasi (simulasi)"])

# ---------------------------------------------------------------- tab deteksi
with tab_det:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", "Musiman" if seasonal_mode else "Lokal")
    c2.metric("Riwayat tersimpan (hari)", f"{hist_days:.1f}")
    c3.metric("Alarm", len(anom))
    c4.metric("Data terakhir (menit lalu)", f"{lag:.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result["edit_count"], mode="lines",
                             name="Aktual", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=result.index, y=result["expected"], mode="lines",
                             name="Baseline ekspektasi", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=anom.index, y=anom["edit_count"], mode="markers",
                             name="Anomali",
                             marker=dict(symbol="x", color="red", size=11)))
    fig.update_layout(xaxis_title="Waktu (UTC)", yaxis_title="Suntingan / menit",
                      hovermode="x unified", height=450)
    st.plotly_chart(fig, use_container_width=True)

    if not anom.empty:
        st.subheader("Konteks anomali")
        ctx = full.loc[anom.index, ["unique_editors", "top_page", "top_page_share", "bot_count"]].copy()
        ctx["bot_share"] = (ctx["bot_count"] / full.loc[anom.index, "edit_count"]).round(2)
        ctx["petunjuk"] = ctx.apply(hint, axis=1)
        ctx.insert(0, "edit_count", anom["edit_count"])
        ctx.insert(1, "mod_z", anom["mod_z"].round(2))
        st.dataframe(ctx.drop(columns="bot_count"), use_container_width=True)
        st.caption("Petunjuk bersifat heuristik (ambang bot >= 60%, halaman teratas >= 25%), "
                   "bukan klasifikasi terlatih.")

    st.subheader("Tabel Kalkulasi")
    st.dataframe(result.round(3).rename(columns={
        "edit_count": "Edit Count", "expected": "Ekspektasi", "rolling_mad": "MAD residual",
        "mod_z": "Mod Z-Score", "flag_raw": "Lewat ambang", "is_anomaly": "Anomaly Status"}),
        use_container_width=True)

# ---------------------------------------------------------------- tab evaluasi
with tab_eval:
    st.write("Lonjakan sintetis disuntikkan ke data nyata untuk mengukur **recall**; "
             "laju alarm pada data asli adalah perkiraan **batas atas** alarm palsu "
             "(data asli mungkin memuat lonjakan sungguhan).")
    duration = st.slider("Durasi lonjakan sintetis (menit)", 1, 5, 2)
    mults = st.multiselect("Pengali lonjakan", [1.25, 1.5, 2.0, 3.0, 5.0], default=[1.5, 2.0, 3.0])
    if st.button("Jalankan simulasi") and mults:
        out = simulate_spikes(cview, det_kw, mults, duration, seasonal=S)
        if out is None:
            st.warning("Data valid terlalu pendek. Perbesar rentang tampilan/data atau kecilkan window.")
        else:
            tbl, fa, nvalid, nspk = out
            a, b = st.columns(2)
            a.metric("Menit beralarm per jam (data asli)", f"{fa:.2f}")
            b.metric("Menit valid / lonjakan disuntik", f"{nvalid} / {nspk}")
            st.dataframe(tbl.style.format({"Recall": "{:.0%}"}), use_container_width=True)
