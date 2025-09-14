# -*- coding: utf-8 -*-
"""
trendline_with_charts_combined_kofn_connected.py

- Per-category sheets: Daily Comments + NormalizedMean（同一sheet，两张图）
- UP/DOWN 连续热区（k-of-n 命中后再做“连续>=N天”过滤），单点不显示
- 图例仅显示“(K-of-N)”规则；标题与单元格不写规则文字
- 热区的点贴 RoC 数值（保留两位小数）
- 统一缺失填补（ffill/zeros/none）保持两边图一致
- 每个视频sheet：写入 videoId/title/category 头信息
- 类目sheet右侧：Top-N (videoId + title) 辅助表
"""

import re
import math
import argparse
from typing import Optional, List
import numpy as np
import pandas as pd

# ---------------- 固定默认路径（如需更改，请用命令行参数，不要改这里） ----------------
DEFAULT_COMMENTS     = None
DEFAULT_COMMENTS_SHT = "comments_kept"
DEFAULT_TITLE_HIER   = None
DEFAULT_VIDEOS       = None
DEFAULT_OUT          = r"C:/Users/desmo/OneDrive/Desktop/dataset/Finalizing/Trendline_output/trendline_with_charts.xlsx"

# ---------------- 全局参数（可用 CLI 覆盖） ----------------
TIMEZONE          = "Asia/Kuala_Lumpur"
DATE_TZ_TO_NAIVE  = True
DUP_POLICY        = "max"       # snapshots 重复行聚合：'max'|'mean'|'last_non_nan'
ENFORCE_MONOTONE  = True        # 视频 commentCount 单调非减

# 阈值与窗口
K_STD        = 1.0              # MAD/STD 的 k 倍数
WIN_DAYS     = 10               # 滚动窗口
BASELINE_WIN = 21               # NormalizedMean 的基线窗口
MIN_DAYS_CAT = 10
MIN_PTS_VID  = 4
TOP_VIDEOS   = 30               # 输出视频图数量上限

# k-of-n（仅体现在图例）
K_HITS = 3
N_WIN  = 5

# RoC / 过滤
EPS_DENOM     = 1.0             # 防除零
MIN_ABS_ROC   = 0.15            # 过滤太小的 RoC
ROC_LABEL_FMT = "{:.2f}"

# 阈值模式
THR_MODE   = "mad"              # 'mad'|'std'|'pct'
THR_PCT_UP = 0.85               # 'pct' 模式上分位
THR_PCT_DN = 0.15               # 'pct' 模式下分位

# 统一缺失值填补：'ffill'|'zeros'|'none'
FILL_STRATEGY = "ffill"

# 只绘制“连续命中”的最小长度（2=至少两天，单点不画；可设3）
MIN_RUN_LEN_FOR_DRAW = 2

# 类目页右侧TOP列表
TOP_TITLES_PER_CAT = 20

# 列名
COL_ID       = "videoId"
COL_L1       = "title_L1"
COL_COMCOUNT = "commentCount"
COL_PUBLISH  = "publishedAt"

# ---------------- 工具函数 ----------------
def sanitize_sheet_name(name: str) -> str:
    s = re.sub(r"[\[\]:\*\?\/\\]", "_", str(name)).strip()
    return (s if s else "Sheet")[:31]

def to_str_vid(v):
    if pd.isna(v): return None
    try:
        f = float(v)
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return str(v)
    except Exception:
        return str(v)

def detect_date_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["date","day","snapshot_date","metric_date"]:
        for c in df.columns:
            if c.lower() == cand:
                return c
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "day" in cl:
            return c
    for c in df.columns:
        if c.lower() == COL_PUBLISH.lower():
            return c
    return None

def to_local_naive(dt_ser: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_ser, errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_convert(TIMEZONE)
    except Exception:
        pass
    if DATE_TZ_TO_NAIVE:
        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            pass
    return dt

def series_to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def apply_fill(s: pd.Series, how: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if how == "ffill":
        return s.ffill()
    if how == "zeros":
        return s.fillna(0.0)
    return s  # 'none'

def compute_roc(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    prev = x.shift(1)
    denom = prev.copy()
    denom[(~np.isfinite(denom)) | (denom <= 0)] = EPS_DENOM
    roc = (x - prev) / denom
    return roc.replace([np.inf, -np.inf], np.nan)

def _win_args(win_days: int):
    return dict(window=win_days, min_periods=max(5, win_days//2))

def _rolling_mean_std(base: pd.Series, win_days: int):
    mu  = base.rolling(**_win_args(win_days)).mean()
    sig = base.rolling(**_win_args(win_days)).std(ddof=0)
    return mu, sig

def _rolling_median_mad(base: pd.Series, win_days: int):
    def _mad(a):
        a = np.asarray(a, dtype=float)
        med = np.nanmedian(a)
        return 1.4826*np.nanmedian(np.abs(a - med))
    med = base.rolling(**_win_args(win_days)).median()
    mad = base.rolling(**_win_args(win_days)).apply(_mad, raw=False)
    return med, mad

def _rolling_quantile_pair(base: pd.Series, win_days: int, p_up: float, p_dn: float):
    center = base.rolling(**_win_args(win_days)).median()
    up  = base.rolling(**_win_args(win_days)).quantile(p_up)
    dn  = base.rolling(**_win_args(win_days)).quantile(p_dn)
    return center, up, dn

def rolling_thresholds(roc: pd.Series, win_days=14, k=1.5):
    base = roc.shift(1)  # exclude today
    if THR_MODE == "mad":
        center, scale = _rolling_median_mad(base, win_days)
        thr_up = center + k*scale
        thr_dn = center - k*scale
    elif THR_MODE == "pct":
        center, thr_up, thr_dn = _rolling_quantile_pair(base, win_days, THR_PCT_UP, THR_PCT_DN)
    else:  # 'std'
        center, scale = _rolling_mean_std(base, win_days)
        thr_up = center + k*scale
        thr_dn = center - k*scale
    return center, thr_up, thr_dn

def k_of_n_mask(flag_series: pd.Series, k_hits=3, n_win=7):
    hits = flag_series.astype(float).rolling(n_win, min_periods=n_win).sum()
    return (hits >= k_hits).fillna(False), hits

def connected_mask(mask: pd.Series, min_len: int = 2):
    """
    只保留“连续 >= min_len”的 True 段；其他（包括单点）设 False
    返回：filtered_mask, runs(list of (start,end))
    """
    m = mask.fillna(False).to_numpy(dtype=bool)
    keep = np.zeros_like(m, dtype=bool)
    runs = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if i - start >= min_len:
                keep[start:i] = True
                runs.append((start, i - 1))
            start = None
    if start is not None and len(m) - start >= min_len:
        keep[start:len(m)] = True
        runs.append((start, len(m) - 1))
    return pd.Series(keep, index=mask.index), runs

def normalized_mean_from_mean(mean_series: pd.Series, baseline_win=21) -> pd.Series:
    base = mean_series.shift(1).rolling(baseline_win, min_periods=max(3, baseline_win//2)).median()
    nm = mean_series / base
    return nm.replace([np.inf, -np.inf], np.nan)

def _build_custom_datalabels(roc_series: pd.Series, hot_mask: pd.Series, fmt: str = ROC_LABEL_FMT):
    """
    仅在 hot_mask=True 的点显示 RoC 数值（自定义数据标签）。
    返回给 XlsxWriter 的 list[dict]。
    """
    labs = []
    roc_vals = pd.to_numeric(roc_series, errors="coerce")
    hot = hot_mask.fillna(False).tolist()
    for i, is_hot in enumerate(hot):
        if is_hot and pd.notna(roc_vals.iloc[i]):
            labs.append({"value": fmt.format(float(roc_vals.iloc[i]))})
        else:
            labs.append({})
    return labs

def _legend_rule() -> str:
    return f" ({K_HITS}-of-{N_WIN})"

# ---------------- 读入 ----------------
def load_comments_table(path: str, sheet: Optional[str]) -> pd.DataFrame:
    if path.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # 找时间列
    need_any_date = None
    for cand in ["publishedAt","commentPublishedAt","updatedAt"]:
        if cand in df.columns:
            need_any_date = cand
            break
    if need_any_date is None:
        raise ValueError("comments 表需要时间列（publishedAt/commentPublishedAt/updatedAt）")

    if "videoId" not in df.columns or "commentId" not in df.columns:
        raise ValueError("comments 表至少需要 videoId 与 commentId")

    df["videoId"] = df["videoId"].map(to_str_vid)
    df = df.dropna(subset=["videoId"]).copy()

    dt = to_local_naive(df[need_any_date])
    df["_ts"]  = dt
    df["date"] = series_to_date(df["_ts"])

    keep = ["videoId","commentId","date"]
    if COL_L1 in df.columns:
        keep.append(COL_L1)
    return df[keep]

def get_video_titles(videos_path: str) -> pd.DataFrame:
    dv = pd.read_csv(videos_path, low_memory=False)
    dv.columns = [c.strip() for c in dv.columns]
    if "videoId" not in dv.columns:
        return pd.DataFrame(columns=["videoId","title"])
    dv["videoId"] = dv["videoId"].map(to_str_vid)

    prefer = ["title","snippet.title","Title","name","video_title"]
    title_col = next((c for c in prefer if c in dv.columns), None)
    if title_col is None:
        dv["title"] = dv["videoId"]
    else:
        dv["title"] = dv[title_col].astype(str).replace({"nan": np.nan}).fillna(dv["videoId"])
    return dv[["videoId","title"]].drop_duplicates("videoId")

def tidy_videos_table(videos_path: str, title_hier: pd.DataFrame) -> pd.DataFrame:
    dv = pd.read_csv(videos_path, low_memory=False)
    dv.columns = [c.strip() for c in dv.columns]
    if "videoId" not in dv.columns or COL_COMCOUNT not in dv.columns:
        raise ValueError("videos 文件需要包含 'videoId' 与 'commentCount'")
    dv["videoId"] = dv["videoId"].map(to_str_vid)

    date_col = detect_date_col(dv)
    if date_col is None:
        raise ValueError("videos 文件需要日期型列（如 date/day/publishedAt 等）")
    dt = to_local_naive(dv[date_col])
    dv["_ts"]  = dt
    dv["date"] = series_to_date(dv["_ts"])

    dv[COL_COMCOUNT] = pd.to_numeric(dv[COL_COMCOUNT], errors="coerce")
    dv.loc[dv[COL_COMCOUNT] < 0, COL_COMCOUNT] = np.nan

    d = dv.merge(title_hier[["videoId",COL_L1]], on="videoId", how="left")
    d = d.dropna(subset=[COL_L1,"date","videoId"]).copy()

    gb = d.groupby(["videoId",COL_L1,"date"], as_index=False)
    if DUP_POLICY == "mean":
        dd = gb.agg(commentCount_clean=(COL_COMCOUNT,"mean"))
    elif DUP_POLICY == "last_non_nan":
        dd = gb.agg(commentCount_clean=(COL_COMCOUNT,"last"))
    else:
        dd = gb.agg(commentCount_clean=(COL_COMCOUNT,"max"))

    if ENFORCE_MONOTONE:
        dd = dd.sort_values(["videoId","date"])
        dd["commentCount_clean"] = dd.groupby("videoId")["commentCount_clean"].cummax()

    dd = dd.dropna(subset=["commentCount_clean"]).copy()
    return dd

# ---------------- 信号构建（含“连续段过滤”） ----------------
def build_signals(df: pd.DataFrame, value_col: str,
                  win_days: int, k_std: float,
                  k_hits: int, n_win: int,
                  min_abs_roc: float,
                  prefix: str):
    """
    生成列：
      RoC_<p>, Ctr_<p>, ThrUp_<p>, ThrDn_<p>,
      FlagUp_<p>, FlagDn_<p>,
      HotUp_<p>, HotDn_<p>,   # 已做“连续>=MIN_RUN_LEN_FOR_DRAW”过滤
      HitsUp_<p>, HitsDn_<p>, # 原始滚动命中数（诊断用）
      MarkedUp_<p>, MarkedDn_<p>  # 仅在连续段里保留值（用于画线）
    """
    roc = compute_roc(df[value_col])
    ctr, up, dn = rolling_thresholds(roc, win_days=win_days, k=k_std)

    flag_up = (roc >= up) & (roc.abs() >= min_abs_roc)
    flag_dn = (roc <= dn) & (roc.abs() >= min_abs_roc)

    hot_up_raw, hits_up = k_of_n_mask(flag_up, k_hits=k_hits, n_win=n_win)
    hot_dn_raw, hits_dn = k_of_n_mask(flag_dn, k_hits=k_hits, n_win=n_win)

    # 只保留连续段（单点不算）
    hot_up, runs_up = connected_mask(hot_up_raw, min_len=MIN_RUN_LEN_FOR_DRAW)
    hot_dn, runs_dn = connected_mask(hot_dn_raw, min_len=MIN_RUN_LEN_FOR_DRAW)

    df[f"RoC_{prefix}"]    = roc
    df[f"Ctr_{prefix}"]    = ctr
    df[f"ThrUp_{prefix}"]  = up
    df[f"ThrDn_{prefix}"]  = dn
    df[f"FlagUp_{prefix}"] = flag_up
    df[f"FlagDn_{prefix}"] = flag_dn
    df[f"HotUp_{prefix}"]  = hot_up
    df[f"HotDn_{prefix}"]  = hot_dn
    df[f"HitsUp_{prefix}"] = hits_up
    df[f"HitsDn_{prefix}"] = hits_dn
    df[f"MarkedUp_{prefix}"] = np.where(hot_up, df[value_col], np.nan)
    df[f"MarkedDn_{prefix}"] = np.where(hot_dn, df[value_col], np.nan)

    return df, runs_up, runs_dn

# ---------------- 类目合并页 ----------------
def write_category_combined_sheet(wb, writer, cat, sub_cmt, sub_vid, segments, summary):
    # union 日期
    dates = []
    if not sub_cmt.empty: dates.append(pd.to_datetime(sub_cmt["date"]))
    if not sub_vid.empty: dates.append(pd.to_datetime(sub_vid["date"]))
    if not dates:
        summary.append({"type":"category","name":cat,"note":"no data"})
        return
    start = min(s.min() for s in dates)
    end   = max(s.max() for s in dates)
    idx = pd.date_range(start, end, freq="D").date
    left = pd.DataFrame({"date": idx})

    # merge daily comments
    if not sub_cmt.empty:
        cmt_aligned = sub_cmt[["date","daily_comments"]].assign(date=pd.to_datetime(sub_cmt["date"]).dt.date)
        left = left.merge(cmt_aligned, on="date", how="left")

    # merge video metrics
    if not sub_vid.empty:
        vid_aligned = (sub_vid[["date","TotalComments","VideoCount","MeanComments","NormalizedMean"]]
                       .assign(date=pd.to_datetime(sub_vid["date"]).dt.date))
        left = left.merge(vid_aligned, on="date", how="left")

    # 统一缺失填补
    for c in ["daily_comments","NormalizedMean","MeanComments","TotalComments","VideoCount"]:
        if c not in left.columns:
            left[c] = np.nan
        left[c] = apply_fill(left[c], FILL_STRATEGY)

    if len(left) < MIN_DAYS_CAT:
        summary.append({"type":"category","name":cat,"note":f"skip <{MIN_DAYS_CAT} days"})
        sh = sanitize_sheet_name(f"Cat_{cat}")
        ws = wb.add_worksheet(sh); writer.sheets[sh] = ws
        tmp = left.copy(); tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
        tmp.to_excel(writer, sheet_name=sh, index=False); return

    # 构建信号
    left, runs_up_d, runs_dn_d = build_signals(left, "daily_comments", WIN_DAYS, K_STD, K_HITS, N_WIN, MIN_ABS_ROC, "Daily")
    left, runs_up_nm, runs_dn_nm = build_signals(left, "NormalizedMean", WIN_DAYS, K_STD, K_HITS, N_WIN, MIN_ABS_ROC, "NM")

    # 写 sheet 基本数据
    sh = sanitize_sheet_name(f"Cat_{cat}")
    ws = wb.add_worksheet(sh); writer.sheets[sh] = ws
    out = left.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    cols = [
        "date",
        # Daily
        "daily_comments","MarkedUp_Daily","MarkedDn_Daily",
        "RoC_Daily","Ctr_Daily","ThrUp_Daily","ThrDn_Daily",
        "FlagUp_Daily","FlagDn_Daily","HotUp_Daily","HotDn_Daily","HitsUp_Daily","HitsDn_Daily",
        # NM
        "NormalizedMean","MarkedUp_NM","MarkedDn_NM",
        "RoC_NM","Ctr_NM","ThrUp_NM","ThrDn_NM",
        "FlagUp_NM","FlagDn_NM","HotUp_NM","HotDn_NM","HitsUp_NM","HitsDn_NM",
        # aux
        "MeanComments","VideoCount","TotalComments"
    ]
    out.to_excel(writer, sheet_name=sh, index=False, startrow=0, startcol=0, columns=cols)

    # chart helpers
    n = len(out)
    def idx_of(c): return cols.index(c)
    def rng(c):    return [sh, 1, idx_of(c), n, idx_of(c)]
    date_rng = rng("date")

    # Chart A: Daily
    if left["daily_comments"].notna().sum() > 0:
        chA = wb.add_chart({"type":"line"})
        chA.add_series({"name": f"{cat} — Daily Comments",
                        "categories": date_rng, "values": rng("daily_comments"),
                        "line":{"width":2.0}})
        # UP overlay
        chA.add_series({
            "name": "UP" + _legend_rule(),
            "categories": date_rng,
            "values": rng("MarkedUp_Daily"),
            "line": {"width": 1.0},
            "marker": {"type": "circle", "size": 6,
                       "border": {"color":"green"}, "fill":{"color":"green"}},
            "data_labels": {"custom": _build_custom_datalabels(left["RoC_Daily"], left["HotUp_Daily"], ROC_LABEL_FMT)},
        })
        # DOWN overlay
        chA.add_series({
            "name": "DOWN" + _legend_rule(),
            "categories": date_rng,
            "values": rng("MarkedDn_Daily"),
            "line": {"width": 1.0},
            "marker": {"type": "circle", "size": 6,
                       "border": {"color":"red"}, "fill":{"color":"red"}},
            "data_labels": {"custom": _build_custom_datalabels(left["RoC_Daily"], left["HotDn_Daily"], ROC_LABEL_FMT)},
        })
        chA.set_title({"name": f"{cat} — Daily Comments"})
        chA.set_x_axis({"name":"Date"})
        chA.set_y_axis({"name":"Daily comments"})
        chA.set_legend({"position":"bottom"})
        ws.insert_chart(1, 8, chA, {"x_scale":1.15,"y_scale":1.15})

        # 记录片段
        for s,e in runs_up_d:
            for i in range(s,e+1):
                segments.append({"scope":"Category","name":cat,"series":"DailyComments","dir":"UP",
                                 "rule":f"{K_HITS}-of-{N_WIN}","date":out.iloc[i]["date"],
                                 "roc":left.iloc[i]["RoC_Daily"],
                                 "thr_up":left.iloc[i]["ThrUp_Daily"],"thr_dn":left.iloc[i]["ThrDn_Daily"]})
        for s,e in runs_dn_d:
            for i in range(s,e+1):
                segments.append({"scope":"Category","name":cat,"series":"DailyComments","dir":"DOWN",
                                 "rule":f"{K_HITS}-of-{N_WIN}","date":out.iloc[i]["date"],
                                 "roc":left.iloc[i]["RoC_Daily"],
                                 "thr_up":left.iloc[i]["ThrUp_Daily"],"thr_dn":left.iloc[i]["ThrDn_Daily"]})

    # Chart B: NormalizedMean
    if left["NormalizedMean"].notna().sum() > 0:
        chB = wb.add_chart({"type":"line"})
        chB.add_series({"name": f"{cat} — NormalizedMean",
                        "categories": date_rng, "values": rng("NormalizedMean"),
                        "line":{"width":2.0}})
        # UP overlay
        chB.add_series({
            "name": "UP" + _legend_rule(),
            "categories": date_rng,
            "values": rng("MarkedUp_NM"),
            "line": {"width": 1.0},
            "marker": {"type": "circle", "size": 6,
                       "border": {"color":"green"}, "fill":{"color":"green"}},
            "data_labels": {"custom": _build_custom_datalabels(left["RoC_NM"], left["HotUp_NM"], ROC_LABEL_FMT)},
        })
        # DOWN overlay
        chB.add_series({
            "name": "DOWN" + _legend_rule(),
            "categories": date_rng,
            "values": rng("MarkedDn_NM"),
            "line": {"width": 1.0},
            "marker": {"type": "circle", "size": 6,
                       "border": {"color":"red"}, "fill":{"color":"red"}},
            "data_labels": {"custom": _build_custom_datalabels(left["RoC_NM"], left["HotDn_NM"], ROC_LABEL_FMT)},
        })
        chB.set_title({"name": f"{cat} — NormalizedMean"})
        chB.set_x_axis({"name":"Date"})
        chB.set_y_axis({"name":"NormalizedMean"})
        chB.set_legend({"position":"bottom"})
        ws.insert_chart(24, 8, chB, {"x_scale":1.15,"y_scale":1.15})

        for s,e in runs_up_nm:
            for i in range(s,e+1):
                segments.append({"scope":"Category","name":cat,"series":"NormalizedMean","dir":"UP",
                                 "rule":f"{K_HITS}-of-{N_WIN}","date":out.iloc[i]["date"],
                                 "roc":left.iloc[i]["RoC_NM"],
                                 "thr_up":left.iloc[i]["ThrUp_NM"],"thr_dn":left.iloc[i]["ThrDn_NM"]})
        for s,e in runs_dn_nm:
            for i in range(s,e+1):
                segments.append({"scope":"Category","name":cat,"series":"NormalizedMean","dir":"DOWN",
                                 "rule":f"{K_HITS}-of-{N_WIN}","date":out.iloc[i]["date"],
                                 "roc":left.iloc[i]["RoC_NM"],
                                 "thr_up":left.iloc[i]["ThrUp_NM"],"thr_dn":left.iloc[i]["ThrDn_NM"]})

    summary.append({"type":"category","name":cat,
                    "note":f"OK days={len(left)} upD={len(runs_up_d)} dnD={len(runs_dn_d)} upNM={len(runs_up_nm)} dnNM={len(runs_dn_nm)}"})

# ---------------- 主流程 ----------------
def build_workbook(comments_path: str, comments_sheet: Optional[str],
                   title_hier_path: str, videos_path: str,
                   out_xlsx: str, top_videos: Optional[int]):

    # 读 title_hier
    th = pd.read_csv(title_hier_path, low_memory=False)
    th.columns = [c.strip() for c in th.columns]
    if "videoId" not in th.columns or COL_L1 not in th.columns:
        raise ValueError("title_hier 必须包含 'videoId' 与 'title_L1'")
    th["videoId"] = th["videoId"].map(to_str_vid)

    # comments
    cmt = load_comments_table(comments_path, comments_sheet)
    cmt = cmt.merge(th[["videoId",COL_L1]], on="videoId", how="left")
    cmt = cmt.dropna(subset=[COL_L1]).copy()

    # videos snapshots（清洗）
    vids = tidy_videos_table(videos_path, th)

    # 类目：来自 comments 的日评论数
    cat_daily_cmt = (cmt.dropna(subset=["videoId","date"])
                        .groupby([COL_L1,"date"], as_index=False)
                        .agg(daily_comments=("commentId","count")))

    # 类目：来自 videos 的 Total/Mean/NormalizedMean
    g = (vids.groupby([COL_L1,"date"], as_index=False)
             .agg(TotalComments=("commentCount_clean","sum"),
                  VideoCount=("videoId","nunique")))
    g["MeanComments"]   = g["TotalComments"] / g["VideoCount"].replace(0,np.nan)
    g["NormalizedMean"] = normalized_mean_from_mean(g["MeanComments"], baseline_win=BASELINE_WIN)
    g = g.sort_values([COL_L1,"date"]).reset_index(drop=True)

    # 视频：日 TotalComments
    vid_daily = (vids.groupby(["videoId",COL_L1,"date"], as_index=False)
                      .agg(TotalComments=("commentCount_clean","max"))
                      .sort_values(["videoId","date"]).reset_index(drop=True))

    # 标题
    titles_df = get_video_titles(videos_path)

    # 统计：每视频评论行数
    video_comments_from_rows = (cmt.groupby("videoId", as_index=False)
                                  .agg(total_comment_rows=("commentId","nunique"))
                                  .merge(th[["videoId",COL_L1]], on="videoId", how="left")
                                  .merge(titles_df, on="videoId", how="left"))

    # 视频总览：首末快照
    tmp = vids.sort_values(["videoId","date"])
    first_vals = tmp.groupby("videoId").first().reset_index()[["videoId",COL_L1,"date","commentCount_clean"]]
    first_vals.rename(columns={"date":"first_date","commentCount_clean":"first_commentCount"}, inplace=True)
    last_vals  = tmp.groupby("videoId").last().reset_index()[["videoId","date","commentCount_clean"]]
    last_vals.rename(columns={"date":"latest_date","commentCount_clean":"latest_commentCount"}, inplace=True)
    video_totals = (first_vals.merge(last_vals, on="videoId", how="left")
                              .merge(titles_df, on="videoId", how="left"))
    video_totals["growth"]      = video_totals["latest_commentCount"] - video_totals["first_commentCount"]
    video_totals["growth_rate"] = np.where(video_totals["first_commentCount"]>0,
                                           video_totals["growth"]/video_totals["first_commentCount"], np.nan)

    cat_totals = video_totals.groupby(COL_L1, as_index=False).agg(
        videos=("videoId","nunique"),
        latest_comments_sum=("latest_commentCount","sum"),
        latest_median=("latest_commentCount","median"),
        latest_mean=("latest_commentCount","mean")
    )

    # ------------- 写 Excel -------------
    writer = pd.ExcelWriter(out_xlsx, engine="xlsxwriter")
    wb = writer.book

    # dump 辅助函数
    def write_df(df: pd.DataFrame, name: str):
        out = df.copy()
        for col in ["date","first_date","latest_date"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
        out.to_excel(writer, sheet_name=name, index=False)

    # 辅助tabs
    write_df(vids.rename(columns={"commentCount_clean":"commentCount"}), "Videos_Clean")
    write_df(cat_daily_cmt, "Comments_Daily")
    write_df(video_totals[["videoId","title",COL_L1,"first_date","first_commentCount","latest_date","latest_commentCount","growth","growth_rate"]],
             "VideoTotals")
    write_df(video_comments_from_rows[["videoId","title", COL_L1, "total_comment_rows"]],
             "VideoCommentCountFromRows")
    write_df(cat_totals, "ByCategoryTotals")

    segments = []
    summary  = []

    # ===== 类目 sheets =====
    all_cats = sorted(set(cat_daily_cmt[COL_L1].unique().tolist()) |
                      set(g[COL_L1].unique().tolist()))
    top_table = video_totals[["videoId","title",COL_L1,"latest_commentCount"]].copy()
    for cat in all_cats:
        sub_cmt = cat_daily_cmt[cat_daily_cmt[COL_L1] == cat].copy()
        sub_vid = g[g[COL_L1] == cat].copy()
        if not sub_cmt.empty:
            sub_cmt["daily_comments"] = pd.to_numeric(sub_cmt["daily_comments"], errors="coerce")
        if not sub_vid.empty:
            for c in ["TotalComments","VideoCount","MeanComments","NormalizedMean"]:
                if c in sub_vid.columns:
                    sub_vid[c] = pd.to_numeric(sub_vid[c], errors="coerce")

        write_category_combined_sheet(wb, writer, cat, sub_cmt, sub_vid, segments, summary)

        # 右侧 Top-N (videoId + title)
        ws = writer.sheets[sanitize_sheet_name(f"Cat_{cat}")]
        tt = top_table[top_table[COL_L1] == cat].copy()
        tt = tt.sort_values("latest_commentCount", ascending=False).head(TOP_TITLES_PER_CAT)
        if not tt.empty:
            start_row, start_col = 0, 24
            ws.write(start_row, start_col, f"Top {TOP_TITLES_PER_CAT} videos in {cat}")
            headers = ["videoId","title","latest_commentCount"]
            for j, h in enumerate(headers):
                ws.write(start_row+1, start_col+j, h)
            for i, (_, r) in enumerate(tt.iterrows(), start=0):
                ws.write(start_row+2+i, start_col+0, str(r["videoId"]))
                ws.write(start_row+2+i, start_col+1, str(r["title"]))
                ws.write_number(start_row+2+i, start_col+2, float(r["latest_commentCount"]))

    # ===== 视频 sheets（Top N） =====
    ranked = video_totals.sort_values("latest_commentCount", ascending=False)
    cand = ranked["videoId"].tolist()
    if top_videos is not None and top_videos > 0:
        cand = cand[:top_videos]

    for vid in cand:
        sub = vid_daily[vid_daily["videoId"]==vid].sort_values("date").reset_index(drop=True)
        if len(sub) < MIN_PTS_VID:
            summary.append({"type":"video","name":vid,"note":f"skip <{MIN_PTS_VID} points"})
            continue
        cat = sub[COL_L1].dropna().iloc[0] if sub[COL_L1].notna().any() else ""

        # 连续日期 & 填补
        idx = pd.date_range(pd.to_datetime(sub["date"].min()),
                            pd.to_datetime(sub["date"].max()), freq="D").date
        sub = sub.set_index("date").reindex(idx).rename_axis("date").reset_index()
        sub["TotalComments"] = apply_fill(sub.get("TotalComments"), FILL_STRATEGY)

        # 信号（prefix 'Vid'）
        sub, runs_up, runs_dn = build_signals(sub, "TotalComments", WIN_DAYS, K_STD, K_HITS, N_WIN, MIN_ABS_ROC, "Vid")

        # 取标题
        title_series = ranked.loc[ranked["videoId"]==vid, "title"]
        title = title_series.iloc[0] if not title_series.empty else vid

        # 写 sheet
        sh = sanitize_sheet_name(f"Vid_{vid}")
        ws = wb.add_worksheet(sh); writer.sheets[sh] = ws
        ws.write(0, 20, "videoId");  ws.write(0, 21, str(vid))
        ws.write(1, 20, "title");    ws.write(1, 21, str(title))
        ws.write(2, 20, "category"); ws.write(2, 21, str(cat))

        tmp = sub.copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
        cols = [
            "date","TotalComments","MarkedUp_Vid","MarkedDn_Vid",
            "RoC_Vid","Ctr_Vid","ThrUp_Vid","ThrDn_Vid",
            "FlagUp_Vid","FlagDn_Vid","HotUp_Vid","HotDn_Vid",
            "HitsUp_Vid","HitsDn_Vid"
        ]
        tmp.to_excel(writer, sheet_name=sh, index=False, startrow=0, startcol=0, columns=cols)

        n = len(tmp)
        def idx_of(c): return cols.index(c)
        def rng(c):    return [sh, 1, idx_of(c), n, idx_of(c)]
        date_rng = rng("date")

        ch = wb.add_chart({"type":"line"})
        ch.add_series({"name": f"{vid} — TotalComments",
                       "categories": date_rng, "values": rng("TotalComments"),
                       "line":{"width":2.0}})
        # UP overlay
        ch.add_series({
            "name": "UP" + _legend_rule(),
            "categories": date_rng,
            "values": rng("MarkedUp_Vid"),
            "line": {"width": 1.0},
            "marker": {"type": "circle", "size": 6,
                       "border": {"color":"green"}, "fill":{"color":"green"}},
            "data_labels": {"custom": _build_custom_datalabels(sub["RoC_Vid"], sub["HotUp_Vid"], ROC_LABEL_FMT)},
        })
        # DOWN overlay
        ch.add_series({
            "name": "DOWN" + _legend_rule(),
            "categories": date_rng,
            "values": rng("MarkedDn_Vid"),
            "line": {"width": 1.0},
            "marker": {"type": "circle", "size": 6,
                       "border": {"color":"red"}, "fill":{"color":"red"}},
            "data_labels": {"custom": _build_custom_datalabels(sub["RoC_Vid"], sub["HotDn_Vid"], ROC_LABEL_FMT)},
        })
        ch.set_title({"name": f"Video {vid} — {title} (Cat: {cat})"})
        ch.set_x_axis({"name":"Date"})
        ch.set_y_axis({"name":"TotalComments"})
        ch.set_legend({"position":"bottom"})
        ws.insert_chart(1, 8, ch, {"x_scale":1.15,"y_scale":1.15})

        # 记录片段
        for s,e in runs_up:
            for i in range(s,e+1):
                segments.append({"scope":"Video","name":vid,"title":str(title),
                                 "series":"TotalComments","dir":"UP","rule":f"{K_HITS}-of-{N_WIN}",
                                 "date":tmp.iloc[i]["date"],"roc":sub.iloc[i]["RoC_Vid"],
                                 "thr_up":sub.iloc[i]["ThrUp_Vid"],"thr_dn":sub.iloc[i]["ThrDn_Vid"]})
        for s,e in runs_dn:
            for i in range(s,e+1):
                segments.append({"scope":"Video","name":vid,"title":str(title),
                                 "series":"TotalComments","dir":"DOWN","rule":f"{K_HITS}-of-{N_WIN}",
                                 "date":tmp.iloc[i]["date"],"roc":sub.iloc[i]["RoC_Vid"],
                                 "thr_up":sub.iloc[i]["ThrUp_Vid"],"thr_dn":sub.iloc[i]["ThrDn_Vid"]})

        summary.append({"type":"video","name":vid,
                        "note":f"OK points={len(sub)} up_runs={len(runs_up)} dn_runs={len(runs_dn)}"})

    # Segments & Summary tabs
    seg_df = pd.DataFrame(segments) if segments else pd.DataFrame(
        columns=["scope","name","title","series","dir","rule","date","roc","thr_up","thr_dn"]
    )
    if not seg_df.empty:
        seg_df["date"] = pd.to_datetime(seg_df["date"]).dt.strftime("%Y-%m-%d")
        seg_df = seg_df.sort_values(["scope","name","series","date"])
    seg_df.to_excel(writer, sheet_name="Segments", index=False)

    pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

    writer.close()
    print(f"[OK] Excel written to: {out_xlsx}")

# ---------------- CLI ----------------

# === Programmatic entry point for combined runner ===
def run_trend(comments: str,
              comments_sheet: str = "comments_kept",
              title_hier: str | None = None,
              videos: str | None = None,
              out: str = "trendline_with_charts.xlsx",
              top_videos: int | None = None):
    """
    Programmatic wrapper to build the trendline workbook.
    It mirrors CLI args but can be called directly from another script.
    """
    return build_workbook(comments, comments_sheet, title_hier, videos, out, top_videos)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", default=DEFAULT_COMMENTS, help="comments Excel/CSV (e.g., daily_scoring_output_*.xlsx)")
    ap.add_argument("--comments_sheet", default=DEFAULT_COMMENTS_SHT, help="sheet name if Excel (default: comments_kept)")
    ap.add_argument("--title_hier", default=DEFAULT_TITLE_HIER, help="video_title_hier CSV (must have videoId,title_L1)")
    ap.add_argument("--videos", default=DEFAULT_VIDEOS, help="videos CSV (must have videoId,commentCount,date-like col)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output .xlsx file path")
    ap.add_argument("--top_videos", type=int, default=TOP_VIDEOS, help="how many top videos to chart; <=0 = ALL")

    # 可调参数（按需）
    ap.add_argument("--k_std", type=float, default=K_STD)
    ap.add_argument("--win_days", type=int, default=WIN_DAYS)
    ap.add_argument("--k_hits", type=int, default=K_HITS)
    ap.add_argument("--n_win", type=int, default=N_WIN)
    ap.add_argument("--min_abs_roc", type=float, default=MIN_ABS_ROC)
    ap.add_argument("--thr_mode", choices=["mad","std","pct"], default=THR_MODE)
    ap.add_argument("--thr_pct_up", type=float, default=THR_PCT_UP)
    ap.add_argument("--thr_pct_dn", type=float, default=THR_PCT_DN)
    ap.add_argument("--fill_strategy", choices=["ffill","zeros","none"], default=FILL_STRATEGY)
    ap.add_argument("--min_run_len", type=int, default=MIN_RUN_LEN_FOR_DRAW)
    args = ap.parse_args()

    # 应用 CLI 覆盖
    K_STD = args.k_std
    WIN_DAYS = args.win_days
    K_HITS = args.k_hits
    N_WIN = args.n_win
    MIN_ABS_ROC = args.min_abs_roc
    THR_MODE = args.thr_mode
    THR_PCT_UP = args.thr_pct_up
    THR_PCT_DN = args.thr_pct_dn
    FILL_STRATEGY = args.fill_strategy
    MIN_RUN_LEN_FOR_DRAW = args.min_run_len

    tv = None if args.top_videos is None or args.top_videos <= 0 else args.top_videos
    build_workbook(args.comments, args.comments_sheet, args.title_hier, args.videos, args.out, tv)
