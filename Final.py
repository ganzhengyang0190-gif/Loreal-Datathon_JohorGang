from approximatelyfinacodemarketing import main as run_scoring
from trendlinescorrrect import run_trend   # ← 确认模块名与文件名完全一致
from genzscore import run_segment
import os

def main():
    outdir = r"C:/Users/desmo/OneDrive/Desktop/dataset/Finalizing/Output"
    comments = r"C:/Users/desmo/OneDrive/Desktop/dataset/Finalizing/Main_input_file/comments120242025.csv"
    videos   = r"C:/Users/desmo/OneDrive/Desktop/dataset/Finalizing/Main_input_file/videos20242025.csv"

    # 1) 评分主流程 —— 返回各类输出路径
    outpaths = run_scoring(comments, videos, outdir)

    # 2) 趋势线（用“评论明细 CSV”，不是评分 Excel）
    trend_out = os.path.join(outdir, "trendline_with_charts.xlsx")
    title_hier_path = outpaths.get("OUTPUT_TITLE_HIER")
    if not (title_hier_path and os.path.exists(title_hier_path)):
        # 分类文件可能不存在；大多数脚本里传 None 会自动把未分类的设为 'Other'
        title_hier_path = None

    run_trend(
        comments=outpaths["OUTPUT_CSV"],      # ✅ 用评论明细 CSV
        comments_sheet=None,                  # ✅ CSV 不需要 sheet
        title_hier=title_hier_path,           # 可能为 None
        videos=videos,
        out=trend_out
    )

    # 3) 分群饼图（同样使用评论明细 CSV）
    seg_out = os.path.join(outdir, "segment_mark_with_pies.xlsx")
    pies_out = run_segment(
        comments   = outpaths["OUTPUT_CSV"],  # ✅ 用评论明细 CSV
        sheet      = None,                    # ✅ CSV 不需要 sheet
        title_hier = title_hier_path,         # 可能为 None
        out        = seg_out
    )

    print("\n=== ALL DONE ===")
    print("Scoring Excel:", outpaths["OUTPUT_XLSX_SCORING"])
    print("Title Hier CSV:", outpaths.get("OUTPUT_TITLE_HIER"))
    print("Trendline XLSX:", trend_out)
    print("Segment pies XLSX:", pies_out)

if __name__ == "__main__":
    main()
