# app.py — Streamlit UI for Automated Data Storytelling (Local LLM via LM Studio)

import os
import io
import hashlib
import pandas as pd
import streamlit as st
from openai import OpenAI
import datavisualizer as dv  # local module

# ---- LM Studio config (env-first) ----
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LMSTUDIO_API_KEY  = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
LMSTUDIO_MODEL    = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")

st.set_page_config(page_title="Automated Data Storytelling", layout="wide")
st.title("📊 Automated Data Storytelling (Local LLM)")

st.markdown(
    "Upload a CSV to automatically generate exploratory charts and an **executive summary** "
    "with **revenue optimization** insights. Uses your local LM Studio model."
)

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    max_tokens = st.slider("Max tokens (summary)", 200, 1000, 500, 50)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)
    st.markdown("---")
    st.caption("LM Studio should be running (Developer → Start Local Server).")

def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()[:12]

def fresh_fig_dir(base_dir: str, run_id: str) -> str:
    d = os.path.join(base_dir, run_id)
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.lower().endswith(".png"):
                try: os.remove(os.path.join(d, f))
                except Exception: pass
    else:
        os.makedirs(d, exist_ok=True)
    return d

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    run_id = file_hash(raw_bytes)
    base_fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    fig_dir = fresh_fig_dir(base_fig_dir, run_id)

    df = pd.read_csv(io.BytesIO(raw_bytes))
    df = dv.try_parse_datetimes(df)

    st.subheader("👀 Preview")
    st.dataframe(df.head(20))

    # === Rich, generalized charts ===
    with st.spinner("Generating charts..."):
        dv.save_missingness_bars(df, fig_dir)
        dv.save_histograms(df, fig_dir, max_plots=12)
        dv.save_correlation_heatmap(df, fig_dir)
        dv.save_scatter_top_correlations(df, fig_dir, max_pairs=6)
        dv.save_boxplots_by_category(df, fig_dir, max_pairs=10)
        dv.save_time_series(df, fig_dir, max_metrics=3)
        dv.save_category_pivots(df, fig_dir, max_pairs=6)

    st.subheader("📈 Exploratory Charts")
    imgs = sorted([f for f in os.listdir(fig_dir) if f.lower().endswith(".png")])
    if imgs:
        cols = st.columns(2)
        for i, img in enumerate(imgs):
            with cols[i % 2]:
                st.image(os.path.join(fig_dir, img), caption=img, use_container_width=True)
    else:
        st.info("No charts were generated (dataset may be too small or non-standard).")

    # === Executive Summary (LM Studio) ===
    st.subheader("🧠 Executive Summary (Local LLM via LM Studio)")
    with st.spinner("Generating executive summary..."):
        prompt = dv.build_exec_prompt(df, fig_dir)
        client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)

        # sanity check model exists
        models = client.models.list()
        ids = [m.id for m in getattr(models, "data", [])]
        if LMSTUDIO_MODEL not in ids:
            st.error(f"Model '{LMSTUDIO_MODEL}' not found on server. Available: {ids}")
        else:
            try:
                resp = client.chat.completions.create(
                    model=LMSTUDIO_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                summary = resp.choices[0].message.content.strip()
                st.success("Summary generated.")
                st.markdown(summary)

                buf = io.BytesIO(summary.encode("utf-8"))
                st.download_button(
                    label="💾 Download Executive Summary (.txt)",
                    data=buf,
                    file_name="executive_summary.txt",
                    mime="text/plain",
                )
            except Exception as e:
                st.error(f"LM Studio summary failed: {e}")
else:
    st.info("⬆️ Upload a CSV to begin.")
