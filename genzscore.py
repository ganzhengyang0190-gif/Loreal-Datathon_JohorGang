# -*- coding: utf-8 -*-
"""
segment_mark_by_category_with_pies_grid.py
Gen Z & Millennials marks + ratio pies (12 per sheet, auto-paginated)

Run (uses defaults below):
  python segment_mark_by_category_with_pies_grid.py
Or override:
  python segment_mark_by_category_with_pies_grid.py \
    --comments "…/daily_scoring_output_20250913_150554.xlsx" --sheet comments_kept \
    --title_hier "…/video_title_hier_20250913_150554.csv" \
    --out "…/segment_mark_results.xlsx"
"""

import os, re, glob, argparse, json, math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ----------------------- DEFAULT PATHS -----------------------
DEFAULT_OUTPUT_DIR = r"C:/Users/desmo/OneDrive/Desktop/dataset/Finalizing/Sentimentals_output"
DEFAULT_COMMENTS   = None
DEFAULT_SHEET      = "comments_kept"
DEFAULT_TITLE_HIER = None
DEFAULT_OUT        = r""  # empty -> auto timestamped output beside detected comments file

# ----------------------- GRID LAYOUT FOR PIE PAGES -----------------------
PIE_GRID_ROWS  = int(os.getenv('PIE_GRID_ROWS',  '3'))   # 3 rows × 4 cols = 12 pies/sheet
PIE_GRID_COLS  = int(os.getenv('PIE_GRID_COLS',  '4'))
PIE_BLOCK_ROWS = int(os.getenv('PIE_BLOCK_ROWS', '18'))  # vertical space per pie block
PIE_BLOCK_COLS = int(os.getenv('PIE_BLOCK_COLS', '10'))  # horizontal space per pie block
PIE_CHART_XS   = float(os.getenv('PIE_CHART_XSCALE', '0.95'))
PIE_CHART_YS   = float(os.getenv('PIE_CHART_YSCALE', '0.95'))

# ----------------------- Scoring knobs -----------------------
THRESH_GENZ = float(os.getenv('THRESH_GENZ', '0.30'))
THRESH_MILL = float(os.getenv('THRESH_MILL', '0.30'))
K_GZ = float(os.getenv('K_GZ', '1.2'))   # saturation factor for Gen Z  (1 - exp(-K * weight_sum))
K_ML = float(os.getenv('K_ML', '1.0'))   # saturation factor for Millennials
TOPN_FEATURES = int(os.getenv('TOPN_SEGMENT_FEATURES', '300'))
PLOT_VIDEO = (os.getenv('PLOT_SEGMENT_BY_VIDEO', '0').lower() in {'1','true','yes','y'})

SKIP_ZERO_DEN = (os.getenv('SKIP_ZERO_DEN', '1').lower() in {'1','true','yes','y'})

# ----------------------- Helpers -----------------------
def read_any(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith('.csv'):
        for enc in ("utf-8","utf-8-sig","utf-16","utf-16le","utf-16","utf-16be","gb18030","cp936","big5","cp950","cp1252","latin1"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except Exception:
                continue
        return pd.read_csv(path, encoding="latin1", low_memory=False)
    else:
        return pd.read_excel(path)

def _latest_in_dir(dir_path: str, patterns: list[str]) -> str:
    paths=[]
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(dir_path, pat)))
    if not paths:
        return ""
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def to_clean_videoid(x):
    if pd.isna(x): return None
    s = str(x).strip()
    try:
        f = float(s.replace(',', ''))
        if np.isfinite(f) and abs(f - int(f)) < 1e-9:
            return str(int(f))
    except Exception:
        pass
    return s

# ----------------------- Lexicons -----------------------
EMOJI_GZ = list("😭💀🥹🫶✨🤌🫣🫠🙃🤯🥳😮‍💨😵‍💫🤝🧠🫡🗿")
SLANG_EN_GZ = [
    r"\bfr\b", r"\bno\s*cap\b", r"\blow\s*key\b", r"\bhigh\s*key\b", r"\bbet\b",
    r"\bslay\b", r"\bate(\s|$)", r"ate and left no crumbs", r"it's\s+giving",
    r"\brizz(er|ler)?\b", r"\bw\s*rizz\b|\bn\s*rizz\b", r"\bdelulu\b",
    r"\bgyatt?\b", r"\bsigma\b", r"\bgoat\b", r"\bvalid\b", r"\bbased\b",
    r"\bmid\b", r"\bcap\b", r"\bratio\b", r"\bick\b", r"ick list",
    r"bombastic\s+side\s+eye|side\s+eye", r"girl\s+(math|dinner)", r"boy\s+math",
    r"\buwu\b", r"\bnpc\b", r"mog(ging)?\b", r"touch\s+grass", r"skibidi",
    r"fanum\s*tax", r"nah\s+he\s+tweakin", r"griddy", r"drip\b|fit\s*check",
    r"\bngl\b", r"\btea\b|spill\s+the\s+tea", r"\bpookie\b", r"mother(ed)?\b",
]
SLANG_MS_GZ = [
    r"\bweh+i*\b", r"\bweii+i*\b", r"\bgais\b", r"\bskrg\b", r"\bmcm\b", r"\bjap\b",
    r"\bcun\b", r"\bpadu\b", r"\bmantap|mantul|mantopp?\b", r"\bsyok\b", r"\bgempak\b",
    r"\bpower\s*gila\b", r"\bsteady\b", r"\bmampus\b", r"\bfyp\b", r"\btrend(ing)?\b",
    r"\bcapcut\b", r"\bviral\b", r"\bbestie\b", r"\bcantiknye|cantiknya\b",
]
SLANG_ZH_GZ = [
    "yyds","绝绝子","emo","冲冲冲","爆改","摆烂","躺平","栓Q","无语子","好家伙",
    "我哭死","绝了","太会了","离谱","离大谱","蚌埠住了","上头","下头","破防",
    "拿捏","会玩","牛哇","牛逼","牛批","逆天","整活","寄","嗑到了","磕到了",
    "嗑cp","乐子","好耶","爷青回","爷青结","内卷","上大分","bking","万能的某宝",
]
TIKTOK_MARKERS = [
    r"\bfyp\b", r"\btiktok\b", r"\bdouyin\b", r"\bcapcut\b", r"\bgrwm\b",
    r"\bpov\b", r"\bhaul\b", r"\bootd\b", r"\brecap\b", r"\bfilter\b",
    r"glow\s*up", r"trend(ing)?", r"sound\s+used", r"template",
]
SCHOOL_LIFE = [
    r"\buni\b", r"\bcollege\b", r"\bcampus\b", r"\bspm\b", r"\bstpm\b", r"\bexam\b",
    r"mid ?term", r"assignment", r"semester", r"hostel", r"dorm", r"class(es)?",
    r"fresh\s*grad", r"intern(ship)?", r"siswazah", r"ptptn",
]
EMOJI_ML = list("😂😅😉👍🙏😍😘😊😁😄😃😆✨🎉🙌💯👌😭")
SLANG_EN_ML = [
    r"\btbh\b", r"\bomg\b", r"\bimho\b", r"\bfomo\b", r"adulting", r"i\s+can't\s+even",
    r"\byaa+ss\b", r"on\s+fleek", r"\btotes\b", r"amazeballs", r"epic\s+fail",
    r"hangry", r"basic\s*af", r"squad\s+goals", r"netflix\s+and\s+chill",
    r"weird\s+flex\b", r"cray\b", r"bae\b", r"brunch", r"wanderlust", r"friyay",
    r"#blessed", r"wine\s*o'?clock", r"doggo|pupper", r"food\s*coma",
]
SLANG_MS_ML = [
    r"\bkerja\b", r"\bmeeting\b", r"\bdeadline\b", r"\bprojek\b", r"\bgaji\b",
    r"\bboss?k?u?\b", r"\bbalik\s*kerja\b", r"\bcari\s*makan\b", r"\brumah\b|\bsewa\b",
    r"\bpinjaman\b|\bloan\b|\bduit\b", r"\bkwsp\b", r"\bptptn\b", r"\basb\b",
    r"\bkahwin\b|\bnikah\b", r"\banak\b|\bbaby\b", r"\bbini\b|\bsuami\b",
    r"\bkereta\b|\bmotor\b|\bminyak\b", r"\bcukai\b|\btax\b", r"\bcuti\b",
]
SLANG_ZH_ML = [
    "上班","加班","老板","工资","薪水","结婚","婚礼","相亲","带娃","育儿","房贷","车贷",
    "通勤","保温杯","中年危机","存钱","理财","KPI",
]
IG_FB_MARKERS = [
    r"\binsta(gram)?\b|\big\b", r"\bfacebook\b|\bfb\b", r"#tbt|#latergram",
    r"boomerang", r"valencia|x-?pro\s*ii|lo-fi|clarendon|lark|juno",
    r"repost", r"link\s*in\s*bio", r"swipe\s*up",
]
WORK_LIFE = [
    r"9\s*[-–]\s*5", r"\bcareer\b", r"\bkpi\b", r"deadline", r"overtime|\bot\b",
    r"mortgage|loan|installment", r"\bsalary\b|\bgaji\b", r"promotion|appraisal",
]
FAMILY_LIFE = [
    r"wedding|married|husband|wife|spouse", r"parent(ing)?|kid(s)?|child|baby|toddler",
    r"preschool|kindergarten", r"家庭|育儿|孩子|宝宝|奶粉",
]

# compile regexes
_EMOJI_GZ_RE = re.compile("|".join(map(re.escape, EMOJI_GZ)))
_EMOJI_ML_RE = re.compile("|".join(map(re.escape, EMOJI_ML)))
_EN_GZ = [re.compile(p, re.I) for p in SLANG_EN_GZ]
_MS_GZ = [re.compile(p, re.I) for p in SLANG_MS_GZ]
_EN_ML = [re.compile(p, re.I) for p in SLANG_EN_ML]
_MS_ML = [re.compile(p, re.I) for p in SLANG_MS_ML]
_TT = [re.compile(p, re.I) for p in TIKTOK_MARKERS]
_SC = [re.compile(p, re.I) for p in SCHOOL_LIFE]
_IGFB = [re.compile(p, re.I) for p in IG_FB_MARKERS]
_WORK = [re.compile(p, re.I) for p in WORK_LIFE]
_FAM  = [re.compile(p, re.I) for p in FAMILY_LIFE]

# ----------------------- Weights -----------------------
W_GZ = {
    'emoji_newschool': 0.25,
    'slang_en_gz':     0.25,
    'slang_ms_gz':     0.15,
    'slang_zh_gz':     0.15,
    'tiktok_markers':  0.20,
    'school_life':     0.10,
    'short_plus_emoji':0.10,
}
W_ML = {
    'emoji_legacy':    0.20,
    'slang_en_ml':     0.25,
    'slang_ms_ml':     0.15,
    'slang_zh_ml':     0.15,
    'ig_fb_markers':   0.20,
    'work_life':       0.15,
    'family_life':     0.15,
}

# ----------------------- Scorers -----------------------
def _count_zh_hits(text: str, vocab: list[str]) -> int:
    t = text or ""
    c = 0
    for kw in vocab:
        if kw and kw in t:
            c += 1
    return c

def _sat_score(hits: dict, weights: dict, k: float) -> float:
    total = 0.0
    for name, cnt in hits.items():
        total += weights.get(name, 0.0) * float(cnt)
    return max(0.0, min(1.0, 1.0 - math.exp(-k * total)))

def genz_mark(text: str) -> tuple[float, dict]:
    if not isinstance(text, str) or not text.strip():
        return 0.0, {}
    t = text.strip()
    hits = {}
    e = len(_EMOJI_GZ_RE.findall(t))
    if e: hits['emoji_newschool'] = e
    en = sum(1 for rx in _EN_GZ if rx.search(t))
    if en: hits['slang_en_gz'] = en
    ms = sum(1 for rx in _MS_GZ if rx.search(t))
    if ms: hits['slang_ms_gz'] = ms
    zh = _count_zh_hits(t, SLANG_ZH_GZ)
    if zh: hits['slang_zh_gz'] = zh
    tt = sum(1 for rx in _TT if rx.search(t))
    if tt: hits['tiktok_markers'] = tt
    sc = sum(1 for rx in _SC if rx.search(t))
    if sc: hits['school_life'] = sc
    tokens = re.findall(r"\w+", t)
    if len(tokens) <= 5 and e:
        hits['short_plus_emoji'] = 1
    score = _sat_score(hits, W_GZ, K_GZ)
    if len(tokens) > 80:
        score *= 0.9
    return score, hits

def millennial_mark(text: str) -> tuple[float, dict]:
    if not isinstance(text, str) or not text.strip():
        return 0.0, {}
    t = text.strip()
    hits = {}
    e = len(_EMOJI_ML_RE.findall(t))
    if e: hits['emoji_legacy'] = e
    en = sum(1 for rx in _EN_ML if rx.search(t))
    if en: hits['slang_en_ml'] = en
    ms = sum(1 for rx in _MS_ML if rx.search(t))
    if ms: hits['slang_ms_ml'] = ms
    zh = _count_zh_hits(t, SLANG_ZH_ML)
    if zh: hits['slang_zh_ml'] = zh
    ig = sum(1 for rx in _IGFB if rx.search(t))
    if ig: hits['ig_fb_markers'] = ig
    wk = sum(1 for rx in _WORK if rx.search(t))
    if wk: hits['work_life'] = wk
    fm = sum(1 for rx in _FAM if rx.search(t))
    if fm: hits['family_life'] = fm
    score = _sat_score(hits, W_ML, K_ML)
    return score, hits

# ----------------------- Pipeline -----------------------
def load_comments_table(path: str, sheet: str | None) -> pd.DataFrame:
    low = path.lower()
    if low.endswith(('.xlsx','.xls')):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = read_any(path)
    df.columns = [c.strip() for c in df.columns]
    for must in ['videoId','commentId','textOriginal']:
        if must not in df.columns:
            raise SystemExit(f"[ERROR] comments missing column: {must}")
    for col in ['videoId','commentId','authorId','channelId','parentCommentId']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({'nan': np.nan, 'none': np.nan, '': np.nan})
    df['videoId'] = df['videoId'].map(to_clean_videoid)
    df = df[df['videoId'].notna()].copy()
    return df

def attach_title_levels(df: pd.DataFrame, title_hier_path: str | None) -> pd.DataFrame:
    if {'title_L1','title_L2'}.issubset(df.columns):
        for c in ['title_L1','title_L2']:
            df[c] = df[c].fillna('Other')
        return df
    if title_hier_path and Path(title_hier_path).exists():
        th = read_any(title_hier_path)
        th.columns = [c.strip() for c in th.columns]
        keep = ['videoId','title_L1','title_L2']
        have = [c for c in keep if c in th.columns]
        if 'videoId' not in have:
            raise SystemExit("[ERROR] title_hier missing videoId column")
        th['videoId'] = th['videoId'].map(to_clean_videoid)
        df = df.merge(th[have], on='videoId', how='left')
    for c in ['title_L1','title_L2']:
        if c not in df.columns:
            df[c] = 'Other'
        df[c] = df[c].fillna('Other')
    return df

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    gz_scores, gz_hits, ml_scores, ml_hits = [], [], [], []
    for txt in df['textOriginal'].astype(str).tolist():
        gs, gh = genz_mark(txt)
        ms, mh = millennial_mark(txt)
        gz_scores.append(gs); gz_hits.append(json.dumps(gh, ensure_ascii=False))
        ml_scores.append(ms); ml_hits.append(json.dumps(mh, ensure_ascii=False))
    out = df.copy()
    out['genz_score'] = gz_scores
    out['mill_score'] = ml_scores
    out['genz_hit'] = (out['genz_score'] >= THRESH_GENZ).astype(int)
    out['mill_hit'] = (out['mill_score'] >= THRESH_MILL).astype(int)
    out['genz_hits'] = gz_hits
    out['mill_hits'] = ml_hits
    return out

def feature_counts(series_json: pd.Series) -> pd.DataFrame:
    cnt = {}
    for js in series_json:
        try:
            d = json.loads(js)
            for k, v in d.items():
                cnt[k] = cnt.get(k, 0) + 1
        except Exception:
            pass
    if not cnt:
        return pd.DataFrame(columns=['feature','count'])
    return (pd.DataFrame([{'feature': k, 'count': v} for k, v in cnt.items()])
              .sort_values('count', ascending=False))

def aggregate_tables(df: pd.DataFrame):
    by_L1 = (df.groupby('title_L1', as_index=False)
               .agg(comments=('commentId','count'),
                    genz_share=('genz_hit','mean'),
                    mill_share=('mill_hit','mean'),
                    genz_score_avg=('genz_score','mean'),
                    mill_score_avg=('mill_score','mean'))
               .sort_values(['genz_share','mill_share','comments'], ascending=[False, False, False]))

    by_L2 = (df.groupby(['title_L1','title_L2'], as_index=False)
               .agg(comments=('commentId','count'),
                    genz_share=('genz_hit','mean'),
                    mill_share=('mill_hit','mean'),
                    genz_score_avg=('genz_score','mean'),
                    mill_score_avg=('mill_score','mean'))
               .sort_values(['title_L1','genz_share','mill_share','comments'], ascending=[True, False, False, False]))

    # Ratios based on *average scores*
    def _add_ratios(dd: pd.DataFrame) -> pd.DataFrame:
        den = (dd['genz_score_avg'] + dd['mill_score_avg']).astype(float)
        if SKIP_ZERO_DEN:
            dd = dd.copy()
            dd = dd[den > 0].copy()
            den = (dd['genz_score_avg'] + dd['mill_score_avg']).astype(float)
        dd['genz_ratio'] = np.where(den > 0, dd['genz_score_avg'] / den, 0.0)
        dd['mill_ratio'] = np.where(den > 0, dd['mill_score_avg'] / den, 0.0)
        return dd

    by_L1 = _add_ratios(by_L1)
    by_L2 = _add_ratios(by_L2)

    by_vid = pd.DataFrame()
    if PLOT_VIDEO:
        by_vid = (df.groupby('videoId', as_index=False)
                    .agg(comments=('commentId','count'),
                         genz_share=('genz_hit','mean'),
                         mill_share=('mill_hit','mean'),
                         genz_score_avg=('genz_score','mean'),
                         mill_score_avg=('mill_score','mean'))
                    .sort_values(['genz_share','mill_share','comments'], ascending=[False, False, False]))
        by_vid = by_vid.merge(df[['videoId','title_L1','title_L2']].drop_duplicates('videoId'), on='videoId', how='left')

    # samples for QA
    gz_top = (df.sort_values('genz_score', ascending=False)
                .head(150)[['videoId','title_L1','title_L2','commentId','textOriginal','genz_score']])
    gz_low = (df.sort_values('genz_score', ascending=True)
                .head(150)[['videoId','title_L1','title_L2','commentId','textOriginal','genz_score']])
    ml_top = (df.sort_values('mill_score', ascending=False)
                .head(150)[['videoId','title_L1','title_L2','commentId','textOriginal','mill_score']])
    ml_low = (df.sort_values('mill_score', ascending=True)
                .head(150)[['videoId','title_L1','title_L2','commentId','textOriginal','mill_score']])

    return by_L1, by_L2, by_vid, gz_top, gz_low, ml_top, ml_low

# ----------------------- Excel Export -----------------------
def export_excel(out_path: str, tables: dict):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        # --- main tables ---
        tables['by_L1'].to_excel(writer, sheet_name='segments_by_L1', index=False)
        tables['by_L2'].to_excel(writer, sheet_name='segments_by_L2', index=False)
        if len(tables.get('by_vid', pd.DataFrame())):
            tables['by_vid'].to_excel(writer, sheet_name='segments_by_video', index=False)
        tables['gz_feat'].to_excel(writer, sheet_name='genz_feature_counts', index=False)
        tables['ml_feat'].to_excel(writer, sheet_name='mill_feature_counts', index=False)
        tables['gz_top'].to_excel(writer, sheet_name='genz_sample_top', index=False)
        tables['gz_low'].to_excel(writer, sheet_name='genz_sample_low', index=False)
        tables['ml_top'].to_excel(writer, sheet_name='mill_sample_top', index=False)
        tables['ml_low'].to_excel(writer, sheet_name='mill_sample_low', index=False)

        book = writer.book

        # --- comparison charts (top20 by comments) ---
        for sheet_name, key in [('L1_chart','by_L1'), ('L2_chart','by_L2')]:
            df = tables[key].copy().sort_values('comments', ascending=False).head(20)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            ch = book.add_chart({'type':'column'})
            ch.add_series({
                'name': 'GenZ share',
                'categories': [sheet_name, 1, 0, len(df), 0],
                'values':     [sheet_name, 1, 2, len(df), 2],
                'y2_axis': True,
            })
            ch.add_series({
                'name': 'Millennials share',
                'categories': [sheet_name, 1, 0, len(df), 0],
                'values':     [sheet_name, 1, 3, len(df), 3],
                'y2_axis': True,
            })
            ch.add_series({
                'name': 'Comments',
                'categories': [sheet_name, 1, 0, len(df), 0],
                'values':     [sheet_name, 1, 1, len(df), 1],
            })
            ch.set_title({'name': f'{sheet_name} — top20'})
            ch.set_y_axis({'name': 'Comments'})
            ch.set_y2_axis({'name': 'Share'})
            ws.insert_chart(0, 6, ch)

        # ===== hidden-data pies laid out in a grid (no small tables on visible pages) =====
        def _write_hidden_data_sheet(df: pd.DataFrame, data_sheet_name: str) -> list[int]:
            """
            Returns list 'starts' where starts[i] is the header row (in data sheet)
            for pie i. Each pie uses 3 rows: header + 2 values (Gen Z, Millennials).
            """
            ds = book.add_worksheet(data_sheet_name)
            ds.hide()
            starts = []
            for i, r in df.iterrows():
                base = i * 3
                starts.append(base)
                ds.write(base,     0, 'Segment'); ds.write(base,     1, 'Value')
                ds.write(base + 1, 0, 'Gen Z');   ds.write_number(base + 1, 1, float(r['genz_ratio']))
                ds.write(base + 2, 0, 'Millennials'); ds.write_number(base + 2, 1, float(r['mill_ratio']))
            return starts

        def _pies_grid_from_hidden(df: pd.DataFrame, label_cols: list[str],
                                   base_sheet: str, data_sheet_name: str):
            if df.empty:
                return
            starts = _write_hidden_data_sheet(df, data_sheet_name)

            per_sheet = PIE_GRID_ROWS * PIE_GRID_COLS
            pages = int(np.ceil(len(df) / per_sheet))
            for p in range(pages):
                chunk = df.iloc[p*per_sheet:(p+1)*per_sheet].reset_index(drop=True)
                sheetname = f"{base_sheet}_pg{p+1}"
                ws = book.add_worksheet(sheetname)
                writer.sheets[sheetname] = ws

                for i, r in chunk.iterrows():
                    global_i = p*per_sheet + i
                    row_idx = i // PIE_GRID_COLS
                    col_idx = i % PIE_GRID_COLS

                    r0 = 1 + row_idx * PIE_BLOCK_ROWS
                    c0 = 0 + col_idx * PIE_BLOCK_COLS

                    title_txt = " • ".join(str(r[c]) for c in label_cols)
                    ws.write(r0, c0, title_txt)

                    dstart = starts[global_i]  # header row in hidden sheet
                    pie = book.add_chart({'type': 'pie'})
                    pie.add_series({
                        'name':       f"{title_txt} — GenZ vs Millennials",
                        'categories': [data_sheet_name, dstart + 1, 0, dstart + 2, 0],
                        'values':     [data_sheet_name, dstart + 1, 1, dstart + 2, 1],
                        'data_labels': {'percentage': True, 'leader_lines': True}
                    })
                    pie.set_title({'name': title_txt})
                    ws.insert_chart(r0 + 1, c0, pie,
                                    {'x_scale': PIE_CHART_XS, 'y_scale': PIE_CHART_YS})

        # ===== build pies =====
        df_l1 = tables['by_L1'][['title_L1','genz_ratio','mill_ratio']].dropna().copy()
        _pies_grid_from_hidden(df_l1, ['title_L1'], 'pies_L1', 'pie_data_L1')

        df_l2 = tables['by_L2'][['title_L1','title_L2','genz_ratio','mill_ratio']].dropna().copy()
        _pies_grid_from_hidden(df_l2, ['title_L1','title_L2'], 'pies_L2', 'pie_data_L2')

# ======================= 新增: run_segment(...) =======================
def run_segment(comments: str | None = DEFAULT_COMMENTS,
                sheet: str = DEFAULT_SHEET,
                title_hier: str | None = DEFAULT_TITLE_HIER,
                out: str | None = DEFAULT_OUT) -> str:
    """
    封装函数（供 Final.py / 其他脚本复用）：
    - comments: daily_scoring_output_*.xlsx 或 comments_with_video_threads_*.csv；若为 None 则自动在 DEFAULT_OUTPUT_DIR 搜索最新
    - sheet: Excel 工作表名（默认 comments_kept）
    - title_hier: video_title_hier_*.csv（可选）
    - out: 输出 Excel 路径；为空则自动命名；若已存在自动加时间戳
    返回：实际写入的 Excel 路径
    """
    cmt_path = comments or _latest_in_dir(
        DEFAULT_OUTPUT_DIR,
        ['daily_scoring_output_*.xlsx', 'comments_with_video_threads_*.csv']
    )
    if not cmt_path:
        raise SystemExit(f"[ERROR] Could not find comments file in {DEFAULT_OUTPUT_DIR}")

    # 读入 + 类目合并 + 打分
    comments_df = load_comments_table(
        cmt_path,
        sheet if cmt_path.lower().endswith(('.xlsx','.xls')) else None
    )
    comments_df = attach_title_levels(comments_df, title_hier or None)
    scored = compute_scores(comments_df)

    # 汇总表
    by_L1, by_L2, by_vid, gz_top, gz_low, ml_top, ml_low = aggregate_tables(scored)
    gz_feat = feature_counts(scored['genz_hits']).head(TOPN_FEATURES)
    ml_feat = feature_counts(scored['mill_hits']).head(TOPN_FEATURES)

    # 输出路径
    if out:
        out_path = out
        if os.path.exists(out_path):
            base, ext = os.path.splitext(out_path)
            out_path = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    else:
        base_dir = os.path.dirname(cmt_path) or DEFAULT_OUTPUT_DIR
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(base_dir, f"segment_mark_with_pies_{ts}.xlsx")

    # 导出
    tables = {
        'by_L1': by_L1,
        'by_L2': by_L2,
        'by_vid': by_vid,
        'gz_feat': gz_feat,
        'ml_feat': ml_feat,
        'gz_top': gz_top,
        'gz_low': gz_low,
        'ml_top': ml_top,
        'ml_low': ml_low,
    }
    export_excel(out_path, tables)
    return out_path
# ======================= 新增: run_segment(...) =======================

# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comments',   default=DEFAULT_COMMENTS, help='daily_scoring_output.xlsx (sheet=comments_kept) OR comments_with_video_threads.csv')
    ap.add_argument('--sheet',      default=DEFAULT_SHEET,    help='sheet name if Excel (default: comments_kept)')
    ap.add_argument('--title_hier', default=DEFAULT_TITLE_HIER, help='optional: video_title_hier_*.csv for L1/L2')
    ap.add_argument('--out',        default=DEFAULT_OUT,      help='output Excel path (default: auto timestamped)')
    args = ap.parse_args()

    # 改为调用封装函数（命令行体验不变）
    out_path = run_segment(
        comments   = args.comments,
        sheet      = args.sheet,
        title_hier = args.title_hier,
        out        = args.out
    )
    print(f"[DONE] Exported: {out_path}")

if __name__ == '__main__':
    main()
