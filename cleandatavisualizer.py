# datavisualizer.py
# Load dataset (Kaggle or local) → exploratory charts → Executive Summary via LM Studio (local)

import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# =========================
# CONFIG (env-first, with safe defaults)
# =========================

USE_LOCAL_CSV = False
# Repo-relative placeholder (no personal path)
LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "local.csv")

KAGGLE_DATASET_ID = "yashdevladdha/uber-ride-analytics-dashboard"
PREFERRED_FILENAME = "ncr_ride_bookings.csv"

USE_LM_STUDIO = True
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_API_KEY  = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_MODEL    = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
LM_TEMPERATURE = float(os.getenv("LM_TEMPERATURE", "0.3"))
LM_MAX_TOKENS  = int(os.getenv("LM_MAX_TOKENS", "500"))

# =========================
# UTILITIES
# =========================

def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def ensure_fig_dir() -> str:
    """Create a 'figures' folder next to this script and return its path."""
    fig_dir = os.path.join(script_dir(), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

def resolve_dataset_path() -> str:
    """
    Returns a path to a CSV:
      - If USE_LOCAL_CSV, uses LOCAL_CSV_PATH.
      - Else, downloads with kagglehub and picks a CSV in the dataset folder.
    """
    if USE_LOCAL_CSV:
        if not os.path.exists(LOCAL_CSV_PATH):
            raise FileNotFoundError(
                f"Local CSV not found:\n{LOCAL_CSV_PATH}\n"
                "Provide a CSV in sample_data/ or set USE_LOCAL_CSV = False."
            )
        return LOCAL_CSV_PATH

    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is not installed. Install it:\n    pip install kagglehub\n"
            "Or set USE_LOCAL_CSV = True and point LOCAL_CSV_PATH to your CSV."
        )

    ds_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    print(f"Dataset downloaded to: {ds_path}")
    print(f"Files in dataset folder: {os.listdir(ds_path)}")

    preferred = os.path.join(ds_path, PREFERRED_FILENAME)
    if os.path.exists(preferred):
        return preferred

    csvs = glob.glob(os.path.join(ds_path, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in dataset folder: {ds_path}")
    print(f"Auto-selecting CSV: {csvs[0]}")
    return csvs[0]

def try_parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to parse columns that look like dates/times."""
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in ["date", "time", "timestamp"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df

def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))

# =========================
# COLUMN DETECTION + HELPERS
# =========================

def detect_columns(df: pd.DataFrame):
    """Infer roles: id, date, numeric, categorical, boolean."""
    date_cols, bool_cols = [], []
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in df.columns:
        if is_datetime(df[c]):
            date_cols.append(c)
        elif df[c].dropna().isin([0, 1, True, False]).all() and df[c].nunique(dropna=True) <= 2:
            bool_cols.append(c)

    likely_id = []
    for c in df.columns:
        nunq = df[c].nunique(dropna=True)
        if len(df) > 0 and nunq / len(df) > 0.9 and not is_datetime(df[c]):
            likely_id.append(c)

    cat_cols = obj_cols.copy()
    for c in num_cols:
        if df[c].nunique(dropna=True) <= 12 and c not in likely_id:
            cat_cols.append(c)

    cat_cols = [c for c in cat_cols if c not in date_cols and c not in bool_cols and c not in likely_id]
    num_cols_clean = [c for c in num_cols if c not in likely_id]

    return {"date": date_cols, "numeric": num_cols_clean, "categorical": cat_cols, "boolean": bool_cols, "id": likely_id}

def top_k_categories(series: pd.Series, k=15):
    vc = series.astype("category").value_counts(dropna=False).head(k)
    vc.index = [("NaN" if (isinstance(x, float) and pd.isna(x)) else str(x)) for x in vc.index]
    return vc

# =========================
# PLOTTING
# =========================

def save_missingness_bars(df: pd.DataFrame, fig_dir: str):
    miss = df.isnull().mean().sort_values(ascending=False)
    if (miss > 0).any():
        fig, ax = plt.subplots(figsize=(10, 5))
        miss[miss > 0].plot(kind="bar", ax=ax)
        ax.set_title("Missing Data Rate by Column")
        ax.set_ylabel("Proportion Missing")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "missingness_rate.png"))
        plt.close(fig)

def save_histograms(df: pd.DataFrame, fig_dir: str, max_plots=12):
    cols = detect_columns(df)["numeric"]
    if not cols:
        print("No numeric columns found for histograms.")
        return
    cols = cols[:max_plots]
    for col in cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        data = df[col].dropna()
        bins = min(50, max(10, int(math.sqrt(max(1, len(data))))))
        ax.hist(data, bins=bins)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col); ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"hist_{safe_name(col)}.png"))
        plt.close(fig)
    print(f"Saved {len(cols)} histogram(s) to {fig_dir}")

def save_boxplots_by_category(df: pd.DataFrame, fig_dir: str, max_pairs=12):
    cols = detect_columns(df)
    num_cols = cols["numeric"]
    cat_cols = cols["categorical"] + cols["boolean"]
    if not num_cols or not cat_cols:
        return
    pairs = []
    for cat in cat_cols:
        vc = top_k_categories(df[cat], k=6)
        for num in num_cols[:6]:
            pairs.append((num, cat, vc.index.tolist()))
    pairs = pairs[:max_pairs]
    for (num, cat, cats) in pairs:
        sub = df[[num, cat]].copy()
        sub[cat] = sub[cat].astype(str)
        sub = sub[sub[cat].isin(cats)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = [sub[sub[cat] == c][num].dropna().values for c in cats]
        ax.boxplot(groups, labels=cats, showfliers=False)
        ax.set_title(f"{num} by {cat}")
        ax.set_xlabel(cat); ax.set_ylabel(num)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"box_{safe_name(num)}_by_{safe_name(cat)}.png"))
        plt.close(fig)

def save_scatter_top_correlations(df: pd.DataFrame, fig_dir: str, max_pairs=6):
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return
    corr = num.corr(method="spearman", numeric_only=True).abs()
    pairs, cols = [], corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((corr.iloc[i, j], cols[i], cols[j]))
    pairs = [p for p in sorted(pairs, reverse=True) if not math.isnan(p[0])]
    for r, x, y in pairs[:max_pairs]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df[x], df[y], s=10, alpha=0.5)
        ax.set_title(f"Scatter: {x} vs {y} (|ρ|≈{r:.2f})")
        ax.set_xlabel(x); ax.set_ylabel(y)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"scatter_{safe_name(x)}_vs_{safe_name(y)}.png"))
        plt.close(fig)

def save_correlation_heatmap(df: pd.DataFrame, fig_dir: str):
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        print("Not enough numeric columns for a correlation heatmap.")
        return
    corr = num.corr(method="spearman", numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("Spearman Correlation Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "correlation_heatmap.png"))
    plt.close(fig)
    print(f"Saved {os.path.join(fig_dir, 'correlation_heatmap.png')}")

def save_time_series(df: pd.DataFrame, fig_dir: str, max_metrics=3):
    cols = detect_columns(df)
    date_cols, num_cols = cols["date"], cols["numeric"]
    if not date_cols:
        print("No datetime-like column found for time-series plots.")
        return
    date_col = date_cols[0]
    ts = df[[date_col] + num_cols].dropna(subset=[date_col]).copy()
    ts = ts.sort_values(date_col).set_index(date_col)

    daily_counts = ts.resample("D").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    daily_counts.plot(ax=ax)
    ax.set_title(f"Records per Day ({date_col})"); ax.set_xlabel("Date"); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "ts_records_per_day.png"))
    plt.close(fig)

    pick = num_cols[:max_metrics] if num_cols else []
    for col in pick:
        fig, ax = plt.subplots(figsize=(10, 4))
        ts[col].resample("D").mean().plot(ax=ax)
        ax.set_title(f"Daily Mean of {col}"); ax.set_xlabel("Date"); ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"ts_daily_mean_{safe_name(col)}.png"))
        plt.close(fig)

    if ts.index.inferred_type in ("datetime64", "datetime64tz"):
        dow = ts.resample("D").size()
        if len(dow) > 0:
            by_dow = dow.groupby(dow.index.dayofweek).mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            by_dow.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(by_dow)]
            by_dow.plot(kind="bar", ax=ax)
            ax.set_title("Avg Records by Day of Week"); ax.set_ylabel("Avg Daily Count")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "ts_dow_avg.png"))
            plt.close(fig)

def save_category_pivots(df: pd.DataFrame, fig_dir: str, max_pairs=6):
    cols = detect_columns(df)
    cats, nums = cols["categorical"], cols["numeric"]
    if len(cats) == 0:
        return

    score_col = None
    priority = ["revenue","amount","sales","price","fare","total","value"]
    for p in priority:
        for c in nums:
            if p in c.lower():
                score_col = c; break
        if score_col: break
    if not score_col and nums:
        score_col = nums[0]

    for cat in cats[:max_pairs]:
        vc = top_k_categories(df[cat], k=8)
        sub = df[df[cat].astype(str).isin(vc.index)][[cat] + ([score_col] if score_col else [])]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8,4))
        counts = sub[cat].astype(str).value_counts().reindex(vc.index)
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Top {cat} by Count")
        ax.set_xlabel(cat); ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"cat_count_{safe_name(cat)}.png"))
        plt.close(fig)

        if score_col:
            fig, ax = plt.subplots(figsize=(8,4))
            means = sub.groupby(cat)[score_col].mean().reindex(vc.index)
            means.plot(kind="bar", ax=ax)
            ax.set_title(f"Average {score_col} by {cat} (Top categories)")
            ax.set_xlabel(cat); ax.set_ylabel(f"Mean {score_col}")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"cat_mean_{safe_name(score_col)}_by_{safe_name(cat)}.png"))
            plt.close(fig)

    if len(cats) >= 2:
        a, b = cats[0], cats[1]
        top_a = top_k_categories(df[a], k=8).index.tolist()
        top_b = top_k_categories(df[b], k=8).index.tolist()
        sub = df[[a, b]].copy()
        sub[a] = sub[a].astype(str); sub[b] = sub[b].astype(str)
        sub = sub[sub[a].isin(top_a) & sub[b].isin(top_b)]
        if not sub.empty:
            pivot = sub.pivot_table(index=a, columns=b, aggfunc="size", fill_value=0)
            fig, ax = plt.subplots(figsize=(10,6))
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
            ax.set_title(f"Count Heatmap: {a} × {b}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"pivot_count_{safe_name(a)}_x_{safe_name(b)}.png"))
            plt.close(fig)

# =========================
# EXECUTIVE SUMMARY via LM STUDIO
# =========================

def build_exec_prompt(df: pd.DataFrame, fig_dir: str) -> str:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    summary_stats = {}
    if num_cols:
        desc = df[num_cols].describe().round(3)
        keep = [r for r in ["count","mean","std","min","max"] if r in desc.index]
        summary_stats = desc.loc[keep].to_dict()

    top_cats = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).head(5)
        vc.index = [("NaN" if pd.isna(x) else str(x)) for x in vc.index]
        top_cats[col] = vc.to_dict()

    try:
        figs = [f for f in os.listdir(fig_dir) if f.lower().endswith(".png")]
        example_figs = figs[:8]
    except Exception:
        example_figs = []

    return f"""
You are a senior data analyst writing for a business executive.
Produce a crisp, decision-oriented summary and concrete revenue optimization ideas.

DATA SNAPSHOT
- Shape: {df.shape}
- Numeric summary (truncated): {summary_stats}
- Top categories (top 5 each): {top_cats}
- Generated figures available: {example_figs}

INSTRUCTIONS
1) 3–5 bullet key findings (demand patterns, revenue drivers, underperforming segments).
2) Call out obvious correlations or seasonality implied by the snapshot.
3) 4–6 actionable recommendations to maximize revenue (pricing, promotions, segmentation, ops efficiency, channel/payment mix).
4) Keep it concise and exec-friendly. State assumptions briefly if needed.
"""

def generate_executive_summary_lmstudio(prompt: str) -> str:
    """Call LM Studio's OpenAI-compatible local server."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install the OpenAI client: pip install openai")

    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
    models = client.models.list()
    model_ids = [m.id for m in getattr(models, "data", [])]
    if LMSTUDIO_MODEL not in model_ids:
        raise RuntimeError(f"Model '{LMSTUDIO_MODEL}' not found. Available: {model_ids}")

    for max_tokens_try in (LM_MAX_TOKENS, max(200, int(LM_MAX_TOKENS * 0.6))):
        resp = client.chat.completions.create(
            model=LMSTUDIO_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens_try,
            temperature=LM_TEMPERATURE,
        )
        if resp and getattr(resp, "choices", None) and resp.choices[0].message.content:
            return resp.choices[0].message.content.strip()
    raise RuntimeError("LM Studio returned an unexpected/empty response.")

# =========================
# MAIN (CLI use)
# =========================

def main():
    print("Current working directory:", os.getcwd())
    csv_path = resolve_dataset_path()
    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path)

    print("\n=== DATASET OVERVIEW ===")
    print(df.head())
    print("\nShape of dataset:", df.shape)
    print("\n=== BASIC STATS ===")
    print(df.describe(include="all"))
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    print("\n=== CORRELATION MATRIX (numeric only) ===")
    print(df.corr(numeric_only=True))
    print("\n=== UNIQUE VALUES PER COLUMN ===")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

    fig_dir = ensure_fig_dir()
    print("Figures will be saved to:", fig_dir)
    df = try_parse_datetimes(df)

    save_missingness_bars(df, fig_dir)
    save_histograms(df, fig_dir, max_plots=12)
    save_correlation_heatmap(df, fig_dir)
    save_scatter_top_correlations(df, fig_dir, max_pairs=6)
    save_boxplots_by_category(df, fig_dir, max_pairs=10)
    save_time_series(df, fig_dir, max_metrics=3)
    save_category_pivots(df, fig_dir, max_pairs=6)

    if USE_LM_STUDIO:
        print("\n=== EXECUTIVE SUMMARY (LM STUDIO) ===")
        prompt = build_exec_prompt(df, fig_dir)
        summary = generate_executive_summary_lmstudio(prompt)
        print(summary)
        out_txt = os.path.join(fig_dir, "executive_summary.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\nSaved executive summary to: {out_txt}")

    print("\nDone.")

if __name__ == "__main__":
    main()
