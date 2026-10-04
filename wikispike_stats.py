"""Fungsi statistik murni (tanpa Streamlit) agar mudah diuji."""
import warnings

import numpy as np
import pandas as pd

MAD_CONSTANT = 0.6745
SQRT_SD = 0.5                       # sd dari sqrt(X) bila X ~ Poisson
MAD_FLOOR = MAD_CONSTANT * SQRT_SD  # lantai MAD pada skala akar kuadrat


def _mad(x: np.ndarray) -> float:
    return np.median(np.abs(x - np.median(x)))


def score(x: pd.Series, window: int, z_thr: float, one_sided: bool = True,
          k: int = 1, m: int = 1):
    """Shifted rolling median/MAD -> Modified Z -> ambang -> aturan k-dari-m."""
    hist = x.shift(1)  # waktu t hanya melihat t-1, t-2, ... (anti masking)
    med = hist.rolling(window, min_periods=window).median()
    mad = hist.rolling(window, min_periods=window).apply(_mad, raw=True)
    z = MAD_CONSTANT * (x - med) / mad.clip(lower=MAD_FLOOR)
    flag = (z > z_thr) if one_sided else (z.abs() > z_thr)
    alarm = flag.astype(int).rolling(m, min_periods=1).sum() >= k
    return med, mad, z, flag, alarm & flag


def seasonal_baseline(y_full: pd.Series, target_index: pd.DatetimeIndex, days: int,
                      half_width: int, same_daytype: bool = True, min_obs: int = 5) -> pd.Series:
    """Median musiman (skala akar kuadrat): nilai pada jam-menit yang sama di `days`
    hari sebelumnya, dengan lingkungan +-`half_width` menit. `y_full` harus berindeks
    reguler per menit (boleh berisi NaN). Bila `same_daytype`, hanya hari kerja
    dibandingkan hari kerja dan akhir pekan dengan akhir pekan."""
    yv = y_full.to_numpy(float)
    n = len(yv)
    p = y_full.index.get_indexer(target_index)
    tw = np.asarray(target_index.dayofweek >= 5)
    cols = []
    for d in range(1, days + 1):
        past_we = np.asarray((target_index - pd.Timedelta(days=d)).dayofweek >= 5)
        same = (past_we == tw) if same_daytype else np.ones(len(p), dtype=bool)
        for j in range(-half_width, half_width + 1):
            q = p - d * 1440 + j
            ok = (q >= 0) & (q < n) & same
            cols.append(np.where(ok, yv[np.clip(q, 0, n - 1)], np.nan))
    arr = np.column_stack(cols)
    cnt = (~np.isnan(arr)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s = np.nanmedian(arr, axis=1)
    s[cnt < min_obs] = np.nan
    return pd.Series(s, index=target_index)


def detect(counts: pd.Series, window: int, z_thr: float, one_sided: bool = True,
           k: int = 1, m: int = 1, seasonal: pd.Series = None) -> pd.DataFrame:
    """Deteksi pada skala sqrt. Bila `seasonal` diberikan, deteksi dilakukan pada
    residual (sqrt(x) - baseline musiman), lalu shifted median/MAD seperti biasa."""
    y = np.sqrt(counts.clip(lower=0))
    x = y - seasonal if seasonal is not None else y
    med, mad, z, flag, alarm = score(x, window, z_thr, one_sided, k, m)
    expected = ((seasonal + med) if seasonal is not None else med).clip(lower=0) ** 2
    return pd.DataFrame({
        "edit_count": counts, "expected": expected, "rolling_mad": mad,
        "mod_z": z, "flag_raw": flag, "is_anomaly": alarm,
    })


def simulate_spikes(counts: pd.Series, det_kwargs: dict, multipliers, duration: int,
                    seasonal: pd.Series = None):
    """Suntik lonjakan (hitungan x pengali selama `duration` menit) pada jarak teratur.
    Return (tabel recall, alarm per jam pada data asli, menit valid, jumlah lonjakan)."""
    window, m = det_kwargs["window"], det_kwargs.get("m", 1)
    base = detect(counts, seasonal=seasonal, **det_kwargs)
    valid = base["mod_z"].notna()
    hours = valid.sum() / 60
    if hours < 1:
        return None
    false_per_hour = base["is_anomaly"].sum() / hours

    gap = window + duration + m + 3
    starts = [s for s in range(window + 1, len(counts) - duration, gap)
              if valid.iloc[s:s + duration].all()]
    if len(starts) < 3:
        return None

    rows = []
    for mult in multipliers:
        inj = counts.copy()
        for s in starts:
            inj.iloc[s:s + duration] = np.round(counts.iloc[s:s + duration].to_numpy() * mult)
        res = detect(inj, seasonal=seasonal, **det_kwargs)
        hit = sum(bool(res["is_anomaly"].iloc[s:s + duration].any()) for s in starts)
        rows.append({"Pengali lonjakan": f"{mult}x", "Lonjakan disuntik": len(starts),
                     "Terdeteksi": hit, "Recall": hit / len(starts)})
    return pd.DataFrame(rows), false_per_hour, int(valid.sum()), len(starts)


def run_detection(counts_all: pd.Series, view_len: int, det_kw: dict, seasonal_cfg: dict = None):
    """Jalankan deteksi pada `view_len` menit terakhir (+ pemanasan).
    seasonal_cfg: dict(days, half_width, same_daytype) atau None untuk mode lokal.
    Return (counts_dipakai, seasonal_atau_None, hasil_penuh, jumlah_baris_pemanasan)."""
    warm = det_kw["window"] + det_kw.get("m", 1) + 2
    if seasonal_cfg is None:
        c = counts_all.iloc[-(view_len + warm):] if view_len else counts_all
        return c, None, detect(c, **det_kw), warm
    tlen = min(len(counts_all), view_len + warm)
    tidx = counts_all.index[-tlen:]
    s = seasonal_baseline(np.sqrt(counts_all.clip(lower=0)), tidx, **seasonal_cfg)
    c = counts_all.loc[tidx]
    return c, s, detect(c, seasonal=s, **det_kw), warm