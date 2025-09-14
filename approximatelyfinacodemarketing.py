# -*- coding: utf-8 -*-
"""
Merged pipeline (v2025-09-13)
=================================
This file merges your original **bigbigtrymainclassify.py** with the more
standardised front-half from **chain_and_score_clean.py** while preserving all
the extra features you built (language filter, sentiment options, hierarchical
title classification, weekly trends, multi-format exports).

Key features retained + added
- Robust CSV reading & ID normalisation (e.g., "10111.0" -> "10111")
- Canonical video meta (title/description/tags/totals) with coverage diagnostics
- M/E/S Score (unchanged math) + optional global/daily threshold
- NEW: ISR metric = (videoLikeCount + commentCount) / (daily_comments + daily_like_sum)
- NEW: Strict meta requirement + unmatched-after-threshold export sheet
- Language screening (en/zh/ms/ta) + mojibake fix
- Sentiment: Local transformers / HF API / VADER fallback (unchanged)
- Hierarchical title categorisation (embed-first → zero-shot fallback)
- Weekly trend CSV + PNG
- Comment-level CSV + Video-level sentiment pivot CSV
- CLI args (override paths) + ENV toggles for behaviour

Run examples (Windows):
  python bigbigtrymainclassify_merged.py \
      --comments "C:/Users/chinf/Downloads/datathon2025/dataset/comments120242025.csv" \
      --videos   "C:/Users/chinf/Downloads/datathon2025/dataset/videos20242025.csv" \
      --outdir   "C:/Users/chinf/Downloads/datathon2025/dataset/output"

ENV toggles (same as before, plus a few new ones):
  RUN_CATEGORISE, RUN_SENTIMENT, RUN_EMBED_LOCAL, USE_LOCAL_ZSC,
  USE_LOCAL_TRANSFORMERS, USE_HF, HF_TOKEN, EMBED_MODEL
  RUN_SCORING_FILTER, MODE_THRESHOLD, SC_LAMBDA, SC_ALPHA,
  SC_X_WEIGHT, SC_Y_WEIGHT, SC_HIGH_LIKE_THRESH, SC_K_SIGMA, SC_P_PERCENTILE
  STRICT_REQUIRE_META (default 1), KEEP_UNMATCHED_SHEET (default 1), PRINT_TOP_MISSING (default 20)
"""

import os, re, csv, time, json, requests, ast, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ================= ENV / SWITCHES =================
RUN_CATEGORISE       = (os.getenv("RUN_CATEGORISE", "1").lower() in {"1","true","yes","y"})
RUN_SENTIMENT        = (os.getenv("RUN_SENTIMENT",  "1").lower() in {"1","true","yes","y"})
RUN_EMBED_LOCAL      = (os.getenv("RUN_EMBED_LOCAL","1").lower() in {"1","true","yes","y"})
USE_LOCAL_ZSC        = (os.getenv("USE_LOCAL_ZSC",  "0").lower() in {"1","true","yes","y"})   # 分类零样本兜底（慢）
USE_LOCAL_TRANSFORMERS = (os.getenv("USE_LOCAL_TRANSFORMERS","0").lower() in {"1","true","yes","y"})
USE_HF               = (os.getenv("USE_HF", "0").lower() in {"1","true","yes","y"})
HF_TOKEN             = os.getenv("HF_TOKEN","")
EMBED_MODEL          = os.getenv("EMBED_MODEL","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ★ 评分系统开关与参数
RUN_SCORING_FILTER   = (os.getenv("RUN_SCORING_FILTER","1").lower() in {"1","true","yes","y"})
MODE_THRESHOLD       = os.getenv("MODE_THRESHOLD","global")  # "global" 或 "daily"
LAMBDA = float(os.getenv("SC_LAMBDA","0.3"))
ALPHA  = float(os.getenv("SC_ALPHA","1.0"))
X_WEIGHT = float(os.getenv("SC_X_WEIGHT","0.7"))
Y_WEIGHT = float(os.getenv("SC_Y_WEIGHT","0.3"))
HIGH_LIKE_THRESH = int(os.getenv("SC_HIGH_LIKE_THRESH","10"))
K_SIGMA = float(os.getenv("SC_K_SIGMA","1.5"))
P_PERCENTILE = float(os.getenv("SC_P_PERCENTILE","0.9"))

# 新：视频元数据严格性与诊断
STRICT_REQUIRE_META  = (os.getenv("STRICT_REQUIRE_META","1").lower() in {"1","true","yes","y"})
KEEP_UNMATCHED_SHEET = (os.getenv("KEEP_UNMATCHED_SHEET","1").lower() in {"1","true","yes","y"})
PRINT_TOP_MISSING    = int(os.getenv("PRINT_TOP_MISSING","20"))

# ================= 默认路径（可被 CLI 覆盖） =================
DEFAULT_COMMENTS_PATH = r"C:/Users/desmo/OneDrive/Desktop/dataset/Finalizing/Main_input_file/comments120242025.csv"
DEFAULT_VIDEOS_PATH   = r"C:/Users/desmo/OneDrive/Desktop/dataset/Finalizing/Main_input_file/videos20242025.csv"
DEFAULT_OUTDIR        = r"C:/Users/desmo/OneDrive/Desktop/dataset/Finalizing/Output"

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ================= 输出路径生成（基于 outdir） =================
def make_outpaths(outdir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    return {
        "OUTPUT_CSV":             rf"{outdir}/comments_with_video_threads_{STAMP}.csv",
        "OUTPUT_VIDEO_SENTIMENT": rf"{outdir}/video_sentiment_summary_{STAMP}.csv",
        "OUTPUT_TITLE_HIER":     rf"{outdir}/video_title_hier_{STAMP}.csv",
        "OUTPUT_TREND_L1":       rf"{outdir}/trend_L1_weekly_{STAMP}.csv",
        "OUTPUT_TREND_PNG":      rf"{outdir}/trend_L1_weekly_{STAMP}.png",
        "OUTPUT_XLSX_SCORING":   rf"{outdir}/daily_scoring_output_{STAMP}.xlsx",
    }

# ================= 语言检测 =================
try:
    from langdetect import detect
except Exception:
    detect = None
try:
    import langid
except Exception:
    langid = None
ALLOW_LANGS = {"en","zh","ms","ta"}
ALIASES = {"zh-cn":"zh","zh-tw":"zh","zh-hans":"zh","zh-hant":"zh"}
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")

def detect_lang_canon(text: str) -> str:
    if not isinstance(text,str) or not text.strip(): return "unknown"
    t = re.sub(r"http\S+|www\.\S+"," ", text)
    t = re.sub(r"\s+"," ", t).strip()
    votes=[]
    if detect is not None:
        try: votes.append(detect(t))
        except Exception: pass
    if langid is not None:
        try:
            c2,_ = langid.classify(t); votes.append(c2)
        except Exception: pass
    if _TAMIL_RE.search(t): votes.append("ta")
    for v in votes:
        v = ALIASES.get(str(v).lower(), str(v).lower())
        if v in ALLOW_LANGS: return v
    for v in votes:
        if str(v).lower().startswith("zh"): return "zh"
    return ALIASES.get(str(votes[0]).lower(), str(votes[0]).lower()) if votes else "unknown"

# ================= 文本清洗 & VADER =================
try:
    from sentiment_utils import apply_vader, fix_mojibake
except Exception:
    apply_vader = None
    def fix_mojibake(x): return (x or "")

# ================= Robust I/O & Canonicalisation (from chain_and_score_clean) =================

def read_any_csv(path: str) -> pd.DataFrame:
    encs = [
        "utf-8","utf-8-sig","utf-16","utf-16le","utf-16be",
        "gb18030","cp936","big5","cp950","cp1252","latin1"
    ]
    last = None
    for e in encs:
        try:
            return pd.read_csv(path, encoding=e, low_memory=False)
        except Exception as ex:
            last = ex
            continue
    raise last

_def_csv_like = {".csv"}

def _read_any(path: str) -> pd.DataFrame:
    low = path.lower()
    if any(low.endswith(ext) for ext in _def_csv_like):
        return read_any_csv(path)
    return pd.read_excel(path)

def to_clean_videoid(x):
    """Normalize IDs: '10111.0' -> '10111', keep strings unchanged."""
    if pd.isna(x): return None
    s = str(x).strip()
    try:
        f = float(s.replace(",", ""))
        if np.isfinite(f) and abs(f - int(f)) < 1e-9:
            return str(int(f))
    except Exception:
        pass
    return s

def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)

def strip_tz_inplace(df_: pd.DataFrame) -> None:
    from pandas.api.types import is_datetime64tz_dtype
    for col in df_.columns:
        try:
            if is_datetime64tz_dtype(df_[col]):
                df_[col] = df_[col].dt.tz_localize(None)
        except Exception:
            pass

def normalize_tags(x) -> str:
    """Uniform text for tags column: list, JSON-like text, or delimited string → ' | ' joined."""
    if x is None or (isinstance(x, float) and pd.isna(x)): return ""
    if isinstance(x, list):
        return " | ".join([str(i).strip() for i in x if str(i).strip()])
    s = str(x).strip()
    if not s or s.lower() in {"nan","none","null"}: return ""
    if s.startswith("[") and s.endswith("]"):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return " | ".join([str(i).strip() for i in val if str(i).strip()])
        except Exception:
            pass
    for sep in ["|", ",", ";"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep)]
            return " | ".join([p for p in parts if p])
    return s

def canonize_videos_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    # remove Excel "Unnamed" columns
    unnamed = [c for c in out.columns if str(c).startswith("Unnamed:")]
    if unnamed: out = out.drop(columns=unnamed)

    if 'videoId' not in out.columns:
        raise RuntimeError(f"[videos] required column 'videoId' missing. Got: {out.columns.tolist()}")

    out['videoId'] = out['videoId'].map(to_clean_videoid)
    out = out[out['videoId'].notna()].copy()

    # prefer canonical names; create if absent
    def rename_first(canon: str, candidates: list[str]):
        cmap = {c.lower(): c for c in out.columns}
        for cand in candidates:
            if cand.lower() in cmap:
                if cmap[cand.lower()] != canon:
                    out.rename(columns={cmap[cand.lower()]: canon}, inplace=True)
                return True
        if canon not in out.columns:
            out[canon] = np.nan
        return False

    rename_first('title', ['title','snippet.title','name','Title'])
    rename_first('description', ['description','desc','snippet.description'])
    rename_first('tags', ['tags','hashtag','topicCategories','topic_categories','keywords'])
    rename_first('videoLikeCount', ['videoLikeCount','videolikecount','likeCount','likes','video_like_count'])
    rename_first('commentCount',   ['commentCount','comments','comment_count'])
    rename_first('viewCount',      ['viewCount','views','view_count'])

    if rename_first('publishedAt', ['publishedAt','published_at']):
        out['videoPublishedAt'] = parse_dt(out['publishedAt'])
        out.drop(columns=['publishedAt'], inplace=True, errors='ignore')
    else:
        if 'videoPublishedAt' not in out.columns: out['videoPublishedAt'] = pd.NaT

    if 'tags' in out.columns:
        out['tags'] = out['tags'].apply(normalize_tags)

    for vnum in ['videoLikeCount','commentCount','viewCount']:
        if vnum in out.columns:
            out[vnum] = pd.to_numeric(out[vnum], errors='coerce')

    keep = ['videoId','title','description','tags','videoLikeCount','commentCount','viewCount','videoPublishedAt']
    out = out[[c for c in keep if c in out.columns]].drop_duplicates('videoId', keep='last')
    return out

# ================= I/O wrappers =================

def load_comments_any(path: str) -> pd.DataFrame:
    df = _read_any(path)
    df.columns = [c.strip() for c in df.columns]
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed: df = df.drop(columns=unnamed)

    for must in ['videoId','commentId','textOriginal','publishedAt']:
        if must not in df.columns:
            raise RuntimeError(f"[comments] missing required column: {must}. Got: {df.columns.tolist()}")

    # types & cleanup
    for col in ['commentId','videoId','parentCommentId','authorId','channelId']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({'nan': np.nan, 'None': np.nan, '': np.nan})
    df['videoId'] = df['videoId'].map(to_clean_videoid)
    df = df[df['videoId'].notna()].copy()

    if 'likeCount' in df.columns:
        df['likeCount'] = pd.to_numeric(df['likeCount'], errors='coerce').fillna(0).astype(int)

    df['publishedAt'] = parse_dt(df['publishedAt'])
    if 'updatedAt' in df.columns:
        df['updatedAt'] = parse_dt(df['updatedAt'])
        df = df.sort_values('updatedAt').drop_duplicates('commentId', keep='last')
    else:
        df = df.drop_duplicates('commentId', keep='last')

    # thread columns
    if 'parentCommentId' not in df.columns:
        df['parentCommentId'] = pd.NA
    df['is_reply'] = df['parentCommentId'].notna()
    df['thread_id'] = np.where(df['is_reply'], df['parentCommentId'], df['commentId'])
    df['thread_depth'] = np.where(df['is_reply'], 1, 0)

    df['date'] = df['publishedAt'].dt.tz_localize(None).dt.date
    return df

# ================= 分类 Taxonomy（含 Vlog/Sports/Tutorial） =================

def _canonical(name: str) -> str: return name.split(" (")[0].strip()

def _prepare_alias_strings(name: str, aliases: list[str]) -> list[str]:
    outs=[]; base=_canonical(name); outs.append(base); outs.append(name); outs.extend([a for a in aliases if a])
    seen=set(); uniq=[]
    for x in outs:
        if x and x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

DEFAULT_L1 = [
    ("Makeup (化妆)", ["makeup","化妆","solek","make up","grwm","get ready with me"]),
    ("Skincare (护肤)", ["skincare","护肤","penjagaan kulit","routine"]),
    ("Haircare (头发护理)", ["hair care","护发","rambut","hairstyle","hair tutorial"]),
    ("Fragrance (香水)", ["fragrance","perfume","香水"]),
    ("Nails (美甲)", ["nails","美甲"]),
    ("Fashion/Style (穿搭)", ["outfit","穿搭","gaya","ootd"]),
    ("ASMR", ["asmr"]),
    ("Vlog (日常/旅行)", ["vlog","daily vlog","travel vlog","study vlog","校园","旅行","生活记录","日常"]),
    ("Sports (运动/体育)", ["sports","sport","workout","健身","运动","足球","篮球","跑步","瑜伽","training","highlight","比赛","集锦"]),
    ("Tutorial/How-to (教程)", ["tutorial","how to","如何","教学","指南","step by step","tips","窍门","讲解"]),
    ("CultureTrend (潮流/趋势)", ["trend","tiktok","viral","潮流"]),
    ("Tools/Devices (工具/仪器)", ["device","工具","brush","仪器"]),
    ("Other", ["other","杂项"])
]
DEFAULT_L2 = {
    "Makeup (化妆)": [
        ("Glass skin look (水光肌)", ["glass skin","水光肌"]),
        ("Complexion (base)", ["底妆","base","teint"]),
        ("Eye makeup", ["eye makeup","眼妆"]),
        ("Lip makeup", ["lip","唇妆"]),
        ("Brow", ["眉","brow"]),
        ("Tools/Brushes", ["刷具","brush"]),
        ("General makeup tutorial", ["教程","化妆教程","makeup tutorial"])],
    "Skincare (护肤)": [
        ("Hydration/Glow (水润光泽)", ["保湿","hydration","glow"]),
        ("Acne care (祛痘)", ["痘痘","acne"]),
        ("Sunscreen/SPF (防晒)", ["防晒","sunscreen","spf"]),
        ("Cleansing (清洁)", ["洁面","cleanser"]),
        ("Anti-aging (抗老)", ["抗老","anti-aging"]),
        ("Brightening (美白提亮)", ["美白","brighten"]),
        ("Mask/Treatment (面膜/护理)", ["面膜","treatment"])],
    "Haircare (头发护理)": [
        ("Color/Dye (染发)", ["染发","dye"]),
        ("Styling (造型)", ["造型","styling"]),
        ("Care/Treatment (护理/修护)", ["护发","treatment"]),
        ("Tools (工具)", ["卷发棒","straightener","hair tool"])],
    "Fragrance (香水)": [
        ("Perfume review", ["香评","review"]),
        ("Scent layering", ["叠加","layering"]),
        ("Cologne", ["cologne"])],
    "Nails (美甲)": [
        ("Manicure", ["指甲护理"]),
        ("Gel/Acrylic", ["光疗","延长甲"]),
        ("Nail art", ["美甲艺术","贴片"])],
    "Fashion/Style (穿搭)": [
        ("Outfit/Style (穿搭风格)", ["穿搭","style","lookbook","ootd"]),
        ("Wardrobe/Haul", ["开箱","haul"]),
        ("Accessories (配饰)", ["配饰","accessories"])],
    "ASMR": [
        ("Makeup ASMR", ["asmr makeup"]),
        ("Skincare ASMR", ["asmr skincare"]),
        ("General ASMR", ["asmr"])],
    "CultureTrend (潮流/趋势)": [
        ("K-beauty", ["k beauty","韩妆"]),
        ("Viral trend", ["viral","爆款"]),
        ("TikTok trend", ["tiktok trend"])],
    "Tools/Devices (工具/仪器)": [
        ("Beauty device (美容仪)", ["美容仪","device"]),
        ("Brush/Applicator (刷具/工具)", ["刷具","applicator"]),
        ("Hair tool (美发工具)", ["吹风机","straightener"])],
    "Vlog (日常/旅行)": [
        ("Daily / Lifestyle", ["daily vlog","日常","生活记录","study vlog","校园","学生党"]),
        ("Travel", ["travel vlog","旅行","trip","旅游"]),
        ("GRWM / Beauty Vlog", ["grwm","get ready with me","化妆vlog"])],
    "Sports (运动/体育)": [
        ("Workout / Training", ["workout","training","健身","训练","打卡","教程"]),
        ("Highlights / Matches", ["highlights","集锦","比赛","match"]),
        ("Tips / How-to", ["技巧","tips","how to","教学"])],
    "Tutorial/How-to (教程)": [
        ("Makeup / Beauty", ["makeup tutorial","化妆教程","护肤教程","skincare routine"]),
        ("General How-to", ["how to","教程","step by step","guide","指南"]),
        ("DIY / Craft", ["diy","手作","craft"])],
    "Other": [("Other", ["other"]) ]
}
DEFAULT_L3 = {
    ("Makeup (化妆)", "Glass skin look (水光肌)"): [
        ("Foundation (粉底液)", ["foundation","粉底液"]),
        ("BB/CC cream", ["bb","cc"]),
        ("Cushion", ["气垫","cushion"]),
        ("Primer", ["妆前","打底"]),
        ("Highlighter", ["高光","highlight"]),
        ("Setting spray", ["定妆喷雾","setting spray"]),
        ("Powder", ["散粉","powder"])],
    ("Makeup (化妆)", "Complexion (base)"): [
        ("Foundation (粉底液)", ["foundation","粉底液"]),
        ("Concealer", ["遮瑕"]),
        ("Primer", ["妆前"]),
        ("BB/CC cream", ["bb","cc"]),
        ("Cushion", ["气垫"]),
        ("Powder", ["散粉"]),
        ("Setting spray", ["定妆"])],
    ("Makeup (化妆)", "Eye makeup"): [
        ("Eyeliner", ["眼线","eyeliner"]),
        ("Mascara", ["睫毛膏","mascara"]),
        ("Eyeshadow", ["眼影","eyeshadow"]),
        ("False lashes", ["假睫毛"])],
    ("Makeup (化妆)", "Lip makeup"): [
        ("Lipstick", ["口红","lipstick"]),
        ("Lip gloss", ["唇蜜"]),
        ("Lip tint", ["唇釉"]),
        ("Lip liner", ["唇线"])],
    ("Skincare (护肤)", "Hydration/Glow (水润光泽)"): [
        ("Moisturizer", ["面霜","moisturizer"]),
        ("Serum", ["精华","serum"]),
        ("Hyaluronic acid", ["玻尿酸","hyaluronic"]),
        ("Toner", ["化妆水","toner"]),
        ("Essence", ["精华水","essence"])],
    ("Skincare (护肤)", "Acne care (祛痘)"): [
        ("Salicylic acid", ["水杨酸"]),
        ("Niacinamide", ["烟酰胺"]),
        ("Benzoyl peroxide", ["过氧化苯甲酰"]),
        ("Retinol", ["维A醇","retinol"])],
    ("Skincare (护肤)", "Sunscreen/SPF (防晒)"): [("Sunscreen", ["防晒"])],
    ("Skincare (护肤)", "Cleansing (清洁)"): [
        ("Cleanser", ["洁面"]), ("Cleansing oil", ["卸妆油"]), ("Micellar water", ["卸妆水"])],
}

def build_taxonomy():
    L1_alias_map = {name: _prepare_alias_strings(name, aliases) for name, aliases in DEFAULT_L1}
    L2_alias_map = {p: {name: _prepare_alias_strings(name, aliases) for name, aliases in items}
                    for p, items in DEFAULT_L2.items()}
    L3_alias_map = {key: {name: _prepare_alias_strings(name, aliases) for name, aliases in items}
                    for key, items in DEFAULT_L3.items()}
    return L1_alias_map, L2_alias_map, L3_alias_map

# ========== 本地嵌入 ==========
_local_embed_model = None

def _ensure_local_embedder():
    global _local_embed_model
    if _local_embed_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[INFO] 加载本地嵌入模型：{EMBED_MODEL} …")
        _local_embed_model = SentenceTransformer(EMBED_MODEL)

def local_embed_batch(texts: list[str]) -> list[list[float]]:
    _ensure_local_embedder()
    vecs = _local_embed_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    return [v.tolist() for v in vecs]

def _cos(a,b): return float(sum(x*y for x,y in zip(a,b)))

def embed_alias_maps(L1_map, L2_map, L3_map):
    all_texts, index = [], []
    for lab, texts in L1_map.items():
        all_texts.extend(texts); index.append(("L1", None, lab, len(texts)))
    for p, m in L2_map.items():
        for lab, texts in m.items():
            all_texts.extend(texts); index.append(("L2", p, lab, len(texts)))
    for key, m in L3_map.items():
        for lab, texts in m.items():
            all_texts.extend(texts); index.append(("L3", key, lab, len(texts)))
    vecs = local_embed_batch(all_texts)
    out_L1,out_L2,out_L3 = ({},{},{})
    pos=0
    for lvl,key,lab,n in index:
        chunk = vecs[pos:pos+n]; pos+=n
        if lvl=="L1":
            out_L1.setdefault(lab, []).extend(chunk)
        elif lvl=="L2":
            out_L2.setdefault(key, {}); out_L2[key].setdefault(lab, []).extend(chunk)
        else:
            out_L3.setdefault(key, {}); out_L3[key].setdefault(lab, []).extend(chunk)
    return out_L1,out_L2,out_L3

def _pick_by_embed(title_vec, cand_vecs, cand_labels):
    best_lab, best_sc = "Other", -1.0
    for lab, vecs in zip(cand_labels, cand_vecs):
        sc = max((_cos(title_vec,v) for v in vecs), default=-1.0)
        if sc>best_sc: best_lab, best_sc = lab, sc
    return best_lab, best_sc

_zsc = None

def _ensure_local_zsc():
    global _zsc
    if _zsc is None:
        from transformers import pipeline
        print("[INFO] 加载本地零样本：MoritzLaurer/mDeBERTa-v3-base-mnli-xnli …")
        _zsc = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

def zsc_top1(text, labels):
    _ensure_local_zsc(); res = _zsc(text, labels, multi_label=False)
    return res['labels'][0], float(res['scores'][0])

def classify_title_hier_local(title, L1_alias_map, L2_alias_map, L3_alias_map, L1_vecs, L2_vecs, L3_vecs,
                              th_L1=0.45, th_L2=0.45, th_L3=0.45):
    try:
        tvec = local_embed_batch([title])[0]
    except Exception:
        return {"L1":"Other","L1_score":0.0,"L2":"Other","L2_score":0.0,"L3":"Other","L3_score":0.0,"path":"Other"}
    L1_labels = list(L1_alias_map.keys())
    L1_cand_vecs = [L1_vecs.get(lab, []) for lab in L1_labels]
    L1_pick, L1_sim = _pick_by_embed(tvec, L1_cand_vecs, L1_labels)
    if L1_sim < th_L1 and USE_LOCAL_ZSC: L1_pick, L1_sim = zsc_top1(title, L1_labels)

    L2_labels = list(L2_alias_map.get(L1_pick, {}).keys())
    if not L2_labels: L2_pick, L2_sim = "Other", 0.0
    else:
        L2_cand_vecs = [L2_vecs.get(L1_pick, {}).get(lab, []) for lab in L2_labels]
        L2_pick, L2_sim = _pick_by_embed(tvec, L2_cand_vecs, L2_labels)
        if L2_sim < th_L2 and USE_LOCAL_ZSC: L2_pick, L2_sim = zsc_top1(title, L2_labels)

    L3_labels = list(L3_alias_map.get((L1_pick, L2_pick), {}).keys())
    if not L3_labels: L3_pick, L3_sim = "Other", 0.0
    else:
        L3_cand_vecs = [L3_vecs.get((L1_pick,L2_pick), {}).get(lab, []) for lab in L3_labels]
        L3_pick, L3_sim = _pick_by_embed(tvec, L3_cand_vecs, L3_labels)
        if L3_sim < th_L3 and USE_LOCAL_ZSC: L3_pick, L3_sim = zsc_top1(title, L3_labels)

    path = " > ".join([_canonical(L1_pick), _canonical(L2_pick), _canonical(L3_pick)])
    return {"L1":_canonical(L1_pick),"L1_score":float(L1_sim),
            "L2":_canonical(L2_pick),"L2_score":float(L2_sim),
            "L3":_canonical(L3_pick),"L3_score":float(L3_sim),
            "path":path}

# ========== 情绪（本地 transformers → HF → VADER） ==========
CARDIFF_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment"
SESSION = requests.Session()
CARDIFF_LABEL_MAP = {"LABEL_0":"negative","LABEL_1":"neutral","LABEL_2":"positive",
                     "negative":"negative","neutral":"neutral","positive":"positive"}

def _parse_hf_sentiment_response(data):
    seq=data
    while isinstance(seq,list) and seq and isinstance(seq[0],list): seq=seq[0]
    if isinstance(seq,list) and seq and isinstance(seq[0],dict):
        best = max(seq, key=lambda x: float(x.get("score",0.0)))
        raw = str(best.get("label","")).upper()
        lab = CARDIFF_LABEL_MAP.get(raw, raw.lower() or "neutral")
        return lab, float(best.get("score",0.0))
    return "neutral", 0.0

def hf_api_sentiment_batch(texts, batch_size=32, initial_max_chars=450, min_max_chars=200, sleep_on_rate=0.5):
    if not HF_TOKEN: raise RuntimeError("HF_TOKEN not set")
    headers={"Authorization": f"Bearer {HF_TOKEN}"}
    n=len(texts); labels=["neutral"]*n; scores=[0.0]*n
    clean=[(t or "").strip() for t in texts]; max_chars=initial_max_chars; i=0
    while i<n:
        j=min(i+batch_size,n)
        chunk=[t[:max_chars] if len(t)>max_chars else t for t in clean[i:j]]
        payload={"inputs": chunk, "options":{"wait_for_model":True,"use_cache":True}}
        try: r=SESSION.post(CARDIFF_URL, headers=headers, json=payload, timeout=120)
        except requests.exceptions.RequestException: time.sleep(sleep_on_rate); continue
        if r.status_code==200:
            data=r.json()
            if isinstance(data,list) and data and isinstance(data[0],dict): data=[data]
            if not (isinstance(data,list) and (len(data)==(j-i))):
                for k,t in enumerate(chunk):
                    rr=SESSION.post(CARDIFF_URL, headers=headers, json={"inputs":t,"options":{"wait_for_model":True}}, timeout=60)
                    if rr.status_code==200: lab,sc=_parse_hf_sentiment_response(rr.json())
                    else: lab,sc="neutral",0.0
                    labels[i+k],scores[i+k]=lab,sc
            else:
                for k,cand_list in enumerate(data):
                    lab,sc=_parse_hf_sentiment_response(cand_list)
                    labels[i+k],scores[i+k]=lab,sc
            print(f"\r  HF Sentiment {j}/{n}", end="", flush=True); i=j; continue
        if r.status_code in (429,503): time.sleep(sleep_on_rate); continue
        if r.status_code==400 and max_chars>min_max_chars:
            max_chars=max(min_max_chars, max_chars//2); continue
        raise RuntimeError(f"HF API status {r.status_code}: {r.text[:200]}")
    print(); return labels,scores

def vader_series(text: str):
    try:
        out = apply_vader(text) if apply_vader else None
        if isinstance(out, dict): comp=float(out.get("compound",0.0))
        elif isinstance(out,(list,tuple)) and len(out)>=1: comp=float(out[0])
        elif isinstance(out,(int,float)): comp=float(out)
        else: comp=0.0
    except Exception: comp=0.0
    lab = "positive" if comp>0.05 else ("negative" if comp<-0.05 else "neutral")
    return pd.Series({"sentiment":lab,"compound":comp})

_local_tok=_local_model=None

def _ensure_local_sentiment_model():
    global _local_tok,_local_model
    if _local_model is not None: return
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _local_tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    _local_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment").eval()

def local_transformers_sentiment_batch(texts, batch_size=16, max_len=256):
    try:
        _ensure_local_sentiment_model(); import torch
        labels,scores=[],[]; mapping={0:"negative",1:"neutral",2:"positive"}
        for i in range(0,len(texts),batch_size):
            batch=[(t or "")[:1000] for t in texts[i:i+batch_size]]
            inputs=_local_tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            with torch.no_grad(): logits=_local_model(**inputs).logits; probs=torch.softmax(logits, dim=-1); sc,idx=probs.max(dim=-1)
            labels.extend([mapping[int(j)] for j in idx]); scores.extend([float(s) for s in sc])
        return labels,scores
    except Exception as e:
        raise RuntimeError(f"local transformers failed: {e}")

# ========== 工具 ==========

def compute_threshold_from_series(scores: pd.Series, k: float, p: float) -> float:
    scores = pd.to_numeric(scores, errors="coerce").dropna().values
    if scores.size == 0: return float("nan")
    mu = float(np.mean(scores)); sigma=float(np.std(scores)); perc=float(np.percentile(scores, p*100))
    return max(mu + k*sigma, perc)

# ========= Coverage probe =========

def build_meta_coverage(comments: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    comm_counts = (comments.groupby('videoId', as_index=False)
                           .agg(comment_rows=('commentId','count')))
    vids = videos[['videoId','title','description','tags','videoLikeCount','commentCount','viewCount']].copy()
    probe = comm_counts.merge(vids, on='videoId', how='left')

    def status_row(r):
        # fully missing in videos CSV
        if pd.isna(r.get('title')) and pd.isna(r.get('videoLikeCount')) and pd.isna(r.get('commentCount')) and pd.isna(r.get('viewCount')):
            return "missing_in_videos_csv"
        # present but what fields are missing?
        missing = []
        for col in ['title','videoLikeCount','commentCount','viewCount']:
            if col in r.index and pd.isna(r[col]):
                missing.append(col)
        if missing:
            return "present_but_missing:" + ",".join(missing)
        return "present_full"

    probe['status'] = probe.apply(status_row, axis=1)
    return probe

# ======== Excel Chart Helpers (embedded charts) ========

def _slug(s: str, maxlen: int = 28) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s) if s else "Other").strip("_")
    return (s[:maxlen]) or "Other"

TREND_MAX_VIDEOS_PER_CATEGORY = int(os.getenv("TREND_MAX_VIDEOS_PER_CATEGORY", "999999"))
TREND_ORDER_BY = os.getenv("TREND_ORDER_BY", "Score")  # or 'total_comments'
SPLIT_SHEETS_BY_CATEGORY = (os.getenv("SPLIT_SHEETS_BY_CATEGORY", "0").lower() in {"1","true","yes","y"})
TREND_TIMEZONE = os.getenv("TREND_TIMEZONE", "Asia/Kuala_Lumpur")


def _build_video_event_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-video event timeline with cumulative count.
    Columns: videoId, publishedAt(ts), cum_comments
    """
    d = df[['videoId','publishedAt']].dropna().copy()
    d = d.sort_values(['videoId','publishedAt'])
    d['cum_comments'] = d.groupby('videoId').cumcount() + 1
    return d


def _build_video_daily_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Daily comment counts per video and normalized momentum ratio.
    ratio(date) = daily(date) / max(daily(all prior dates))
    First day ratio defined as 0 (no prior peak).
    Columns: videoId, date, daily, prior_peak, ratio
    """
    g = (df.groupby(['videoId','date'], as_index=False)
            .agg(daily=('commentId','count'))
            .sort_values(['videoId','date']))
    g['prior_peak'] = g.groupby('videoId')['daily'].apply(lambda s: s.shift(1).cummax())
    g['ratio'] = g['daily'] / g['prior_peak']
    g['ratio'] = g['ratio'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return g


def _write_trend_sheets_with_charts(writer, comments_kept: pd.DataFrame, df_videos: pd.DataFrame):
    """Create Excel-embedded charts for:
       - For each video (grouped by L1/L2), Chart A: cumulative comment arrivals over time (x = each comment timestamp)
       - For each video (grouped by L1/L2), Chart B: normalized daily momentum ratio
    """
    book = writer.book

    # Ensure classification columns exist
    for col in ['title_L1','title_L2']:
        if col not in comments_kept.columns:
            comments_kept[col] = 'Other'
        comments_kept[col] = comments_kept[col].fillna('Other')

    # Timezone alignment (optional): convert timestamps to local zone before charting
    df_t = comments_kept.copy()
    try:
        df_t['publishedAt'] = pd.to_datetime(df_t['publishedAt'], utc=True, errors='coerce').dt.tz_convert(TREND_TIMEZONE).dt.tz_localize(None)
    except Exception:
        df_t['publishedAt'] = pd.to_datetime(df_t['publishedAt'], errors='coerce')

    # Precompute series
    ev_series = _build_video_event_series(df_t)
    daily_ratio = _build_video_daily_ratio(df_t)

    def _make_container_sheets(prefix: str):
        if SPLIT_SHEETS_BY_CATEGORY:
            # We'll create sheets on-the-fly per category
            return None
        # Single consolidated sheet per prefix
        sheet = writer.book.add_worksheet(prefix)
        return {prefix: sheet}

    for level, col in [("L1","title_L1"),("L2","title_L2")]:
        sheet_map = _make_container_sheets(f"Trend_{level}")

        # videos sorted per category by TREND_ORDER_BY
        # Build a helper table of video-level ordering metrics
        vid_metrics = (comments_kept.groupby('videoId', as_index=False)
                        .agg(total_comments=('commentId','count'),
                             any_score=('Score','max'))
                       )
        order_key = 'any_score' if TREND_ORDER_BY.lower().startswith('score') else 'total_comments'

        for cat, df_cat in comments_kept.groupby(col):
            if SPLIT_SHEETS_BY_CATEGORY:
                ws_name = f"{level}_{_slug(cat)}"
                sheet = book.add_worksheet(ws_name)
                cur_row = 0
            else:
                sheet = list(sheet_map.values())[0]
                # category separator title
                cur_row = sheet.dim_rowmax + 2 if sheet.dim_rowmax is not None else 0
            # Write a header for the category
            sheet.write(cur_row, 0, f"Category {level}: {cat}"); cur_row += 1
            # pick videos in this category
            vids = sorted(df_cat['videoId'].dropna().unique().tolist())
            # sort by metric
            vm = vid_metrics[vid_metrics['videoId'].isin(vids)].sort_values(order_key, ascending=False)
            vids_sorted = vm['videoId'].tolist()
            if TREND_MAX_VIDEOS_PER_CATEGORY and len(vids_sorted) > TREND_MAX_VIDEOS_PER_CATEGORY:
                vids_sorted = vids_sorted[:TREND_MAX_VIDEOS_PER_CATEGORY]

            for vid in vids_sorted:
                # small meta
                title = df_videos.loc[df_videos['videoId']==vid, 'title'].dropna().astype(str).iloc[0] if (df_videos is not None and 'title' in df_videos.columns and (df_videos['videoId']==vid).any()) else str(vid)
                sheet.write(cur_row, 0, f"videoId: {vid}")
                sheet.write(cur_row, 1, f"title: {title}"); cur_row += 1

                # Event timeline table
                ev = ev_series[ev_series['videoId']==vid][['publishedAt','cum_comments']].copy()
                ev_start = cur_row
                sheet.write_row(cur_row, 0, ["publishedAt","cum_comments"]); cur_row += 1
                for _, r in ev.iterrows():
                    sheet.write_datetime(cur_row, 0, r['publishedAt'])
                    sheet.write_number(cur_row, 1, float(r['cum_comments']))
                    cur_row += 1
                ev_end = cur_row - 1

                # Chart A: cumulative arrival
                ch1 = book.add_chart({'type':'line'})
                ch1.add_series({
                    'name': f'Cumulative comments',
                    'categories': [sheet.name, ev_start+1, 0, ev_end, 0],
                    'values':     [sheet.name, ev_start+1, 1, ev_end, 1],
                })
                ch1.set_title({'name': 'Comment arrivals over time'})
                ch1.set_x_axis({'name': 'Time'})
                ch1.set_y_axis({'name': 'Cum. comments'})
                sheet.insert_chart(ev_start, 3, ch1, {'x_scale': 1.2, 'y_scale': 1.0})
                cur_row += 1

                # Daily ratio table
                dr = daily_ratio[daily_ratio['videoId']==vid][['date','daily','prior_peak','ratio']].copy()
                dr_start = cur_row
                sheet.write_row(cur_row, 0, ["date","daily","prior_peak","ratio"]); cur_row += 1
                for _, r in dr.iterrows():
                    # date written as datetime
                    if not isinstance(r['date'], pd.Timestamp):
                        dtv = pd.to_datetime(r['date'])
                    else:
                        dtv = r['date']
                    sheet.write_datetime(cur_row, 0, dtv)
                    sheet.write_number(cur_row, 1, float(r['daily']))
                    sheet.write_number(cur_row, 2, float(0.0 if pd.isna(r['prior_peak']) else r['prior_peak']))
                    sheet.write_number(cur_row, 3, float(r['ratio']))
                    cur_row += 1
                dr_end = cur_row - 1

                # Chart B: normalized daily momentum
                ch2 = book.add_chart({'type':'line'})
                ch2.add_series({
                    'name': f'Ratio = daily / prior_peak',
                    'categories': [sheet.name, dr_start+1, 0, dr_end, 0],
                    'values':     [sheet.name, dr_start+1, 3, dr_end, 3],
                })
                ch2.set_title({'name': 'Normalized daily momentum'})
                ch2.set_x_axis({'name': 'Date'})
                ch2.set_y_axis({'name': 'Ratio (0~1+)', 'major_unit': 0.2})
                sheet.insert_chart(dr_start, 3, ch2, {'x_scale': 1.2, 'y_scale': 1.0})
                cur_row += 2

    # Optional: Weekly top-5 chart in workbook instead of PNG
    # Build weekly trend using existing logic if available
    return

# ================= 主流程 =================

def main(comments_path: str, videos_path: str, outdir: str):
    outpaths = make_outpaths(outdir)
    # 1) 加载（更健壮的读取 & 规范化）
    df_comments = load_comments_any(comments_path)
    df_videos_raw = _read_any(videos_path)
    df_videos = canonize_videos_df(df_videos_raw)

    # 1.1 Join diagnostics（打印交集/缺失）
    set_c, set_v = set(df_comments['videoId']), set(df_videos['videoId'])
    inter = set_c & set_v
    print(f"[INFO] comments unique videoIds: {len(set_c):,}")
    print(f"[INFO] videos   unique videoIds: {len(set_v):,}")
    print(f"[INFO] intersection videoIds    : {len(inter):,}")
    missing_ids = sorted(list(set_c - set_v))
    if PRINT_TOP_MISSING and missing_ids:
        print(f"[INFO] top {min(PRINT_TOP_MISSING, len(missing_ids))} missing videoIds:", missing_ids[:PRINT_TOP_MISSING])

    # 2) 语言筛选 + 清洗（对 comments 文本）
    print("语言筛选（仅保留：English / 中文 / Melayu / தமிழ்）…")
    df_comments["lang"] = df_comments["textOriginal"].astype(str).map(detect_lang_canon)
    before_n=len(df_comments); df_comments=df_comments[df_comments["lang"].isin(ALLOW_LANGS)].copy(); after_n=len(df_comments)
    print(f"已筛除 {before_n-after_n} 行非目标语言；保留 {after_n} 行。")
    df_comments['text_clean'] = df_comments['textOriginal'].astype(str).map(fix_mojibake)
    df_comments['text_en'] = df_comments['text_clean']

    # 3) 情绪分析（对清洗文本）
    if RUN_SENTIMENT:
        texts = df_comments['text_clean'].astype(str).tolist()
        print("SENTIMENT MODE CHECK:",
              "USE_LOCAL_TRANSFORMERS=", USE_LOCAL_TRANSFORMERS,
              "USE_HF=", USE_HF, "HF_TOKEN_SET=", bool(HF_TOKEN))
        try:
            if USE_LOCAL_TRANSFORMERS:
                print("[MODE] 情绪：本地 transformers（离线）…")
                labs,scs = local_transformers_sentiment_batch(texts)
            elif USE_HF and HF_TOKEN:
                print("[MODE] 情绪：HF Inference API（联网）…")
                labs,scs = hf_api_sentiment_batch(texts)
            else:
                print("[MODE] 情绪：本地 VADER（离线）…")
                vs = df_comments['text_clean'].astype(str).apply(vader_series)
                labs,scs = vs['sentiment'].tolist(), vs['compound'].tolist()
            df_comments['sentiment']=labs; df_comments['compound']=scs
        except Exception as e:
            print(f"[WARN] 高级情绪失败，降级到 VADER：{e}")
            vs = df_comments['text_clean'].astype(str).apply(vader_series)
            df_comments['sentiment']=vs['sentiment']; df_comments['compound']=vs['compound']
    else:
        df_comments['sentiment']='neutral'; df_comments['compound']=0.0
        print("[INFO] 跳过情绪分析。")

    # 4) 日聚合 (M/E/S) based on comments + merge video totals before thresholding
    daily = (
        df_comments.groupby(['videoId','date'], as_index=False)
                   .agg(daily_comments=('commentId','count'),
                        daily_like_sum=('likeCount','sum'))
                   .sort_values(['videoId','date'])
    )

    # M
    daily['prev_comments'] = daily.groupby('videoId')['daily_comments'].shift(1).fillna(0)
    daily['M'] = np.log((daily['daily_comments']+1)/(daily['prev_comments']+1))

    # E：log_mean（日均 log1p like）+ high_prop（视频全期高赞占比）
    tmp_like = df_comments[['videoId','date','likeCount']].copy()
    tmp_like['log1p_like'] = np.log1p(tmp_like['likeCount'])
    log_mean_daily = tmp_like.groupby(['videoId','date'], as_index=False).agg(log_mean=('log1p_like','mean'))
    daily = daily.merge(log_mean_daily, on=['videoId','date'], how='left'); daily['log_mean']=daily['log_mean'].fillna(0.0)

    high_prop = (
        df_comments.assign(is_high=(df_comments['likeCount']>=HIGH_LIKE_THRESH).astype(int))
                   .groupby('videoId', as_index=False)
                   .agg(high_cnt=('is_high','sum'), total_cnt=('commentId','count'))
    )
    high_prop['high_prop'] = np.where(high_prop['total_cnt']>0, high_prop['high_cnt']/high_prop['total_cnt'], 0.0)
    daily = daily.merge(high_prop[['videoId','high_prop']], on='videoId', how='left'); daily['high_prop']=daily['high_prop'].fillna(0.0)
    daily['E'] = X_WEIGHT*daily['log_mean'] + Y_WEIGHT*daily['high_prop']

    # S：用当前 comment-level 的视频级情绪
    vs = (df_comments.groupby(['videoId','sentiment'], as_index=False)
                        .size().pivot(index='videoId', columns='sentiment', values='size')
                        .fillna(0).reset_index())
    for c in ['positive','negative','neutral']:
        if c not in vs.columns: vs[c]=0
    vs['total'] = vs['positive'] + vs['negative'] + vs['neutral']
    vs['S'] = (vs['positive'] + 0.5*vs['neutral'] + ALPHA) / (vs['total'] + 2*ALPHA)
    daily = daily.merge(vs[['videoId','S']], on='videoId', how='left'); daily['S']=daily['S'].fillna(0.5)

    # Score
    daily['Score'] = daily['M'] * (1.0 + LAMBDA*daily['E']) * daily['S']

    # Merge video meta (title/desc/tags + totals) BEFORE thresholding, for diagnostics
    daily = daily.merge(
        df_videos[['videoId','title','description','tags','videoLikeCount','commentCount','viewCount']],
        on='videoId', how='left'
    ).rename(columns={'title':'video_title','description':'video_description','tags':'video_tags'})

    # meta presence: require at least one of totals to exist to call it "matched"
    daily['meta_status'] = np.where(
        daily[['videoLikeCount','commentCount','viewCount']].notna().any(axis=1),
        'matched', 'missing_in_videos'
    )

    # ISR = (videoLikeCount + commentCount) / (daily_comments + daily_like_sum)
    video_tot = daily['videoLikeCount'] + daily['commentCount']   # remains NaN if meta missing
    comment_inter = daily['daily_comments'].astype(float) + daily['daily_like_sum'].astype(float)
    den = comment_inter.replace(0, np.nan)
    daily['ISR'] = video_tot / den

    # 阈值过滤
    if RUN_SCORING_FILTER:
        if MODE_THRESHOLD=="global":
            TH = compute_threshold_from_series(daily['Score'], k=K_SIGMA, p=P_PERCENTILE)
            print(f"📌 Global Threshold: {TH:.6f}" if pd.notna(TH) else "📌 Global Threshold: NaN")
            daily_kept = daily[daily['Score']>=TH].copy() if pd.notna(TH) else daily.copy()
        else:
            thresholds = (daily.groupby('date')['Score']
                               .apply(lambda x: compute_threshold_from_series(x, k=K_SIGMA, p=P_PERCENTILE))
                               .reset_index().rename(columns={'Score':'Threshold'}))
            daily_kept = (daily.merge(thresholds, on='date', how='left')
                               .loc[lambda d: d['Score']>=d['Threshold']]
                               .drop(columns=['Threshold'])
                               .copy())
        print("保留 video×day 数量：", len(daily_kept))
    else:
        daily_kept = daily.copy()
        print("[INFO] 未启用评分过滤（RUN_SCORING_FILTER=0），全部保留。")

    # Enforce meta presence in final outputs
    unmatched = pd.DataFrame()
    if STRICT_REQUIRE_META:
        unmatched = daily_kept[daily_kept['meta_status'] != 'matched'].copy()
        daily_kept = daily_kept[daily_kept['meta_status'] == 'matched'].copy()

    # 评分结果导出（Excel 多表）
    scored_videos = daily_kept.copy()
    strip_tz_inplace(scored_videos)

    os.makedirs(outdir, exist_ok=True)
    with pd.ExcelWriter(outpaths['OUTPUT_XLSX_SCORING'], engine="xlsxwriter", datetime_format='yyyy-mm-dd hh:mm:ss') as writer:
        cols = [
            'videoId','date','video_title','video_tags','video_description',
            'Score','M','E','S','ISR','log_mean','high_prop','daily_comments','daily_like_sum',
            'viewCount','videoLikeCount','commentCount'
        ]
        cols = [c for c in cols if c in scored_videos.columns]
        sv = scored_videos[cols].copy()
        sv.rename(columns={'viewCount':'video_view_total','videoLikeCount':'video_like_total','commentCount':'video_comment_total'}, inplace=True)
        sv.sort_values(['Score','video_like_total','video_view_total'], ascending=[False, False, False]).to_excel(writer, sheet_name="scored_videos", index=False)

        # diagnostics
        probe = build_meta_coverage(df_comments, df_videos)
        summary = (probe.groupby('status', as_index=False)
                         .agg(videoIds=('videoId','count'), total_comment_rows=('comment_rows','sum'))
                         .sort_values('videoIds', ascending=False))
        # extra coverage tables
        missing_meta = (probe[probe['status']=='missing_in_videos_csv']
                        .sort_values('comment_rows', ascending=False))
        partial_meta = (probe[probe['status'].str.startswith('present_but_missing')]
                        .sort_values('comment_rows', ascending=False))

        summary.to_excel(writer, sheet_name='diagnostics', index=False)
        probe.to_excel(writer, sheet_name='video_meta_coverage', index=False)
        missing_meta.to_excel(writer, sheet_name='missing_video_meta', index=False)
        partial_meta.to_excel(writer, sheet_name='partial_video_meta', index=False)
        if KEEP_UNMATCHED_SHEET and len(unmatched):
            unmatched.to_excel(writer, sheet_name='unmatched_after_threshold', index=False)

        # === NEW: Embedded charts for per-video timelines & normalized momentum ===
        _write_trend_sheets_with_charts(writer, comments_kept, df_videos)

    print(f"✓ 已导出评分Excel（含嵌入图表）：{outpaths['OUTPUT_XLSX_SCORING']}")

    # ★ comments_kept: 在评分通过的 (videoId,date) 上保留原 comments 行（含 sentiment & thread）
    key_cols = ['videoId','date','Score','M','E','S','ISR','videoLikeCount','commentCount','viewCount','video_title','video_tags','video_description']
    key = scored_videos[key_cols].copy()
    comments_kept = df_comments.merge(key, on=['videoId','date'], how='inner')
    strip_tz_inplace(comments_kept)

    # ===== 之后部分照旧（分类/趋势/导出） =====

    # 5) 分类（仅对筛后的视频）
    titles_hier = pd.DataFrame()
    if RUN_CATEGORISE and RUN_EMBED_LOCAL:
        try:
            keep_vids = sorted(comments_kept['videoId'].dropna().astype(str).unique().tolist())
            vids_titles = df_videos[df_videos['videoId'].astype(str).isin(keep_vids)][['videoId','title']].dropna().drop_duplicates('videoId')
            print(f"[MODE] 分类：本地嵌入，相似度 → {len(vids_titles)} 个视频 …")
            L1_map,L2_map,L3_map = build_taxonomy()
            L1_vecs,L2_vecs,L3_vecs = embed_alias_maps(L1_map,L2_map,L3_map)
            titles_hier = vids_titles.copy()
            L1s,L1sco,L2s,L2sco,L3s,L3sco,paths = [],[],[],[],[],[],[]
            for t in titles_hier['title'].astype(str):
                out = classify_title_hier_local(t, L1_map,L2_map,L3_map, L1_vecs,L2_vecs,L3_vecs, 0.45,0.45,0.45)
                L1s.append(out["L1"]); L1sco.append(out["L1_score"])
                L2s.append(out["L2"]); L2sco.append(out["L2_score"])
                L3s.append(out["L3"]); L3sco.append(out["L3_score"])
                paths.append(out["path"])
            titles_hier["title_L1"]=L1s; titles_hier["title_L1_score"]=L1sco
            titles_hier["title_L2"]=L2s; titles_hier["title_L2_score"]=L2sco
            titles_hier["title_L3"]=L3s; titles_hier["title_L3_score"]=L3sco
            titles_hier["title_path"]=paths
            titles_hier.to_csv(outpaths['OUTPUT_TITLE_HIER'], index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, escapechar='\\')
            print(f"✓ 已导出（视频标题层级分类）：{outpaths['OUTPUT_TITLE_HIER']}")
            comments_kept = comments_kept.merge(titles_hier[['videoId','title_L1','title_L1_score','title_L2','title_L2_score','title_L3','title_L3_score','title_path']], on='videoId', how='left')
        except Exception as e:
            print(f"[WARN] 分类失败：{e}")

    # 6) 周趋势线（筛后数据）
    # 保留 CSV 导出；不再生成 PNG，而是将图形转移到 Excel（见 _write_trend_sheets_with_charts）
    ts_col = 'publishedAt' if 'publishedAt' in comments_kept.columns else None
    if ts_col:
        tmp = comments_kept.copy()
        tmp['ts'] = pd.to_datetime(tmp[ts_col], utc=True, errors='coerce')
        tmp = tmp.dropna(subset=['ts'])
        tmp['week'] = tmp['ts'].dt.to_period('W-MON').dt.start_time
        if 'title_L1' not in tmp.columns: tmp['title_L1']='Other'
        tmp['title_L1'] = tmp['title_L1'].fillna('Other')
        trend = (tmp.groupby(['week','title_L1'], as_index=False)
                    .agg(total_comments=('commentId','count'),
                         videos=('videoId','nunique'),
                         positive=('sentiment', lambda s:(s=='positive').sum()),
                         neutral =('sentiment', lambda s:(s=='neutral').sum()),
                         negative=('sentiment', lambda s:(s=='negative').sum()))
                    .sort_values(['title_L1','week']))
        trend['pos_rate'] = (trend['positive'] / trend['total_comments'].replace(0,np.nan)).fillna(0)
        trend['total_ma4'] = trend.groupby('title_L1')['total_comments'].transform(lambda x: x.rolling(4, min_periods=1).mean())
        trend['wow_growth'] = trend.groupby('title_L1')['total_comments'].pct_change().replace([np.inf,-np.inf], np.nan)
        trend.to_csv(outpaths['OUTPUT_TREND_L1'], index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"✓ 已导出（周趋势 CSV：按 L1）：{outpaths['OUTPUT_TREND_L1']}")
    else:
        print("[WARN] 无可用时间列，无法生成趋势线（CSV）。")

    # 7) 导出评论级 CSV（筛后）
    export_cols = [
        'videoId','video_title','video_tags','video_description',
        'commentId','parentCommentId','is_reply','thread_id','thread_depth',
        'authorId','publishedAt','likeCount','textOriginal','lang','text_en','compound','sentiment',
        'date','Score','M','E','S','ISR',
        'videoLikeCount','commentCount','viewCount'
    ]
    export_cols = [c for c in export_cols if c in comments_kept.columns]
    comments_kept[export_cols].to_csv(outpaths['OUTPUT_CSV'], index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"✓ 已导出（评论级 + 情绪 + 评分 + 分类/元数据）：{outpaths['OUTPUT_CSV']}")

    # 8) 视频聚合情绪（筛后）
    if 'videoId' in comments_kept.columns and 'sentiment' in comments_kept.columns:
        pivot = (comments_kept.groupby(['videoId','sentiment'], as_index=False)
                          .size().pivot(index='videoId', columns='sentiment', values='size')
                          .fillna(0).astype(int).reset_index())
        for col in ['positive','neutral','negative']:
            if col not in pivot.columns: pivot[col]=0
        title_map2 = df_videos[['videoId','title']].dropna().drop_duplicates() if {'videoId','title'}.issubset(df_videos.columns) else None
        if title_map2 is not None:
            pivot = pivot.merge(title_map2.rename(columns={'title':'video_title'}), on='videoId', how='left')
        pivot['total_comments'] = pivot['positive'] + pivot['neutral'] + pivot['negative']
        cols_order = ['videoId','video_title','positive','neutral','negative','total_comments']
        other_cols = [c for c in pivot.columns if c not in cols_order]
        pivot = pivot[cols_order + other_cols].sort_values(['positive','total_comments'], ascending=[False,False])
        pivot.to_csv(outpaths['OUTPUT_VIDEO_SENTIMENT'], index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"✓ 已导出（视频聚合情绪）：{outpaths['OUTPUT_VIDEO_SENTIMENT']}")
    else:
        print("[WARN] 缺少 videoId 或 sentiment 列，无法生成视频聚合情绪统计。")

    return outpaths

# ========= CLI =========
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comments", type=str, default=DEFAULT_COMMENTS_PATH, help="Path to comments csv/excel")
    ap.add_argument("--videos",   type=str, default=DEFAULT_VIDEOS_PATH,   help="Path to videos csv/excel")
    ap.add_argument("--outdir",   type=str, default=DEFAULT_OUTDIR,        help="Directory to store outputs")
    args = ap.parse_args()
    main(args.comments, args.videos, args.outdir)

    
