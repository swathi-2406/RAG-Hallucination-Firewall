"""
app.py
RAG Hallucination Firewall — Streamlit Dashboard

Run with:
    streamlit run app.py
"""

import sys
import logging
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    INDEX_DIR, GROQ_API_KEY,
    ENTROPY_THRESHOLD, JSD_THRESHOLD, NLI_THRESHOLD,
)

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Hallucination Firewall",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

    :root {
        --bg: #0d0f14;
        --surface: #161a23;
        --surface2: #1e2330;
        --border: #2a3040;
        --accent: #00d4ff;
        --accent2: #7c3aed;
        --green: #22c55e;
        --yellow: #f59e0b;
        --red: #ef4444;
        --text: #e2e8f0;
        --muted: #64748b;
        --font-mono: 'JetBrains Mono', monospace;
        --font-sans: 'Inter', sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        color: var(--text);
        font-family: var(--font-sans);
    }

    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff22, #7c3aed22);
        border: 1px solid var(--accent);
        color: var(--accent);
        font-family: var(--font-mono);
        font-weight: 700;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00d4ff44, #7c3aed44);
        box-shadow: 0 0 20px #00d4ff33;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
    }

    .metric-card h4 {
        color: var(--muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0 0 0.4rem 0;
        font-family: var(--font-mono);
    }

    .metric-card .val {
        font-size: 2rem;
        font-weight: 700;
        font-family: var(--font-mono);
        color: var(--text);
        line-height: 1;
    }

    .risk-low    { color: var(--green) !important; }
    .risk-medium { color: var(--yellow) !important; }
    .risk-high   { color: var(--red) !important; }

    .stage-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-family: var(--font-mono);
        font-weight: 700;
        margin-right: 0.4rem;
    }
    .badge-pass { background: #22c55e22; color: var(--green); border: 1px solid var(--green); }
    .badge-fail { background: #ef444422; color: var(--red); border: 1px solid var(--red); }

    .answer-box {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 0.95rem;
        line-height: 1.7;
        margin: 1rem 0;
    }

    .chunk-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
        font-size: 0.82rem;
        font-family: var(--font-mono);
        color: var(--muted);
    }

    .chunk-card .src {
        color: var(--accent);
        font-size: 0.7rem;
        margin-bottom: 0.3rem;
    }

    .header-logo {
        font-family: var(--font-mono);
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.02em;
    }

    .subhead {
        color: var(--muted);
        font-size: 0.8rem;
        font-family: var(--font-mono);
        margin-top: -0.3rem;
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-family: var(--font-mono) !important;
        border-radius: 6px !important;
    }

    [data-testid="stSelectbox"] > div > div {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }

    [data-testid="stTab"] { font-family: var(--font-mono); }

    .stDataFrame { background: var(--surface) !important; }

    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 8px;
    }

    .stSlider > div { color: var(--text); }
    label { color: var(--muted) !important; font-size: 0.8rem !important; }

    h1, h2, h3 { font-family: var(--font-mono); }
    h1 { color: var(--accent); }
    h2 { color: var(--text); font-size: 1rem !important; letter-spacing: 0.05em; }
    h3 { color: var(--text); font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ── Load FAISS index ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Load the FAISS index once and cache it across sessions."""
    try:
        from src.retrieval.retriever import load_index
        return load_index()
    except FileNotFoundError:
        return None


# ── Helper: Risk gauge ────────────────────────────────────────────────────────
def make_gauge(score: float, label: str) -> go.Figure:
    color = "#22c55e" if score < 0.3 else ("#f59e0b" if score < 0.6 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": color, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#64748b", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e2330",
            "bordercolor": "#2a3040",
            "steps": [
                {"range": [0, 30],  "color": "rgba(34,197,94,0.09)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.09)"},
                {"range": [60, 100],"color": "rgba(239,68,68,0.09)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": round(score * 100, 1),
            },
        },
        title={"text": label, "font": {"size": 11, "color": "#64748b", "family": "JetBrains Mono"}},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
    )
    return fig


# ── Helper: Stage bar chart ───────────────────────────────────────────────────
def make_stage_bars(s1, s2, s3) -> go.Figure:
    stages = ["Stage 1<br>Entropy", "Stage 2<br>JSD", "Stage 3<br>NLI"]
    scores = [s1, s2, s3]
    thresholds = [ENTROPY_THRESHOLD, JSD_THRESHOLD, NLI_THRESHOLD]
    colors = [
        "#22c55e" if s < t else "#ef4444"
        for s, t in zip(scores, thresholds)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stages, y=scores,
        marker_color=colors,
        marker_line_color="#2a3040",
        marker_line_width=1,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont={"family": "JetBrains Mono", "size": 11, "color": "#e2e8f0"},
        name="Score",
    ))

    # Threshold lines
    for i, (t, stage) in enumerate(zip(thresholds, stages)):
        fig.add_shape(
            type="line",
            x0=i - 0.4, x1=i + 0.4,
            y0=t, y1=t,
            line=dict(color="#f59e0b", width=2, dash="dot"),
        )

    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        yaxis=dict(
            range=[0, 1],
            gridcolor="#2a3040",
            tickfont={"family": "JetBrains Mono", "size": 9},
        ),
        xaxis=dict(tickfont={"family": "JetBrains Mono", "size": 10}),
        showlegend=False,
        bargap=0.4,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="header-logo">🔥 RAG Firewall</div>', unsafe_allow_html=True)
    st.markdown('<div class="subhead">Hallucination Detection System</div>', unsafe_allow_html=True)
    st.divider()

    # API Key check
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not set in .env")
        st.code("cp .env.example .env\n# Then add your Groq API key", language="bash")

    # Index status
    index_path = INDEX_DIR / "faiss_store"
    if index_path.exists() and index_path.is_dir():
        st.success("✅ FAISS index loaded")
    else:
        st.warning("⚠️ No FAISS index found")
        st.code("python scripts/ingest_docs.py", language="bash")

    st.divider()
    st.markdown("**⚙️ Pipeline Settings**")

    run_entropy = st.toggle("Stage 1: Entropy", value=True)
    run_jsd     = st.toggle("Stage 2: JSD", value=True)
    run_nli     = st.toggle("Stage 3: NLI", value=True)

    st.divider()
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 5)
    mmr_div = st.slider("MMR Diversity", 0.0, 1.0, 0.3, 0.05,
                        help="0 = max diversity, 1 = max relevance")

    st.divider()
    st.markdown("**📖 About**")
    st.caption(
        "Three-stage hallucination detection:\n"
        "1. Semantic entropy (5 LLM samples)\n"
        "2. Jensen-Shannon divergence\n"
        "3. DeBERTa NLI cross-check"
    )


# ── Main Layout ───────────────────────────────────────────────────────────────
tab_query, tab_dashboard, tab_logs = st.tabs([
    "🔍 Query",
    "📊 Dashboard",
    "📋 Query Logs",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUERY
# ══════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.markdown("## QUERY INTERFACE")

    # Load index
    vs = load_vectorstore()

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_area(
            "Enter your question about AI/ML",
            placeholder="e.g. What is retrieval-augmented generation and how does it reduce hallucinations?",
            height=80,
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡ RUN", use_container_width=True)

    if run_btn and query.strip():
        if not vs:
            st.error("FAISS index not loaded. Run `python scripts/ingest_docs.py` first.")
        elif not GROQ_API_KEY:
            st.error("Add your GROQ_API_KEY to the .env file first.")
        else:
            with st.spinner("Running pipeline..."):
                from src.retrieval.retriever import retrieve, get_context_string
                from src.hallucination.firewall import run_firewall
                from src.evaluation.metrics import compute_all_metrics
                from src.evaluation.logger import log_query

                # Retrieval
                t0 = time.perf_counter()
                chunks, ret_latency = retrieve(query, vs, top_k=top_k, diversity=mmr_div)
                chunks_text = [c.page_content for c in chunks]

                # Firewall
                result = run_firewall(
                    query, chunks,
                    run_entropy=run_entropy,
                    run_jsd=run_jsd,
                    run_nli=run_nli,
                )

                # Metrics
                metrics = compute_all_metrics(query, result["answer"], chunks_text)

                # Log
                log_query(query, result["answer"], result, metrics, ret_latency)

                st.session_state.last_result = {
                    "result": result,
                    "metrics": metrics,
                    "ret_latency": ret_latency,
                }
                st.session_state.query_count += 1

    # ── Display result ────────────────────────────────────────────────────────
    if st.session_state.last_result:
        r = st.session_state.last_result["result"]
        m = st.session_state.last_result["metrics"]
        risk = r["composite_risk_score"]
        label = r["risk_label"]
        risk_cls = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[label]

        st.divider()

        # Answer
        st.markdown("**📝 ANSWER**")
        st.markdown(
            f'<div class="answer-box">{r["answer"]}</div>',
            unsafe_allow_html=True,
        )

        # Risk + Stages
        col_gauge, col_stages, col_meta = st.columns([2, 3, 2])

        with col_gauge:
            st.plotly_chart(
                make_gauge(risk, f"{label} RISK"),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_stages:
            s1 = r["stage1_entropy"].get("score", 0)
            s2 = r["stage2_jsd"].get("score", 0)
            s3 = r["stage3_nli"].get("score", 0)
            st.plotly_chart(
                make_stage_bars(s1, s2, s3),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with col_meta:
            st.markdown('<div class="metric-card"><h4>Context Precision</h4>'
                        f'<div class="val">{m["context_precision"]:.2f}</div></div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><h4>Answer Faithfulness</h4>'
                        f'<div class="val">{m["answer_faithfulness"]:.2f}</div></div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="metric-card"><h4>Total Latency</h4>'
                        f'<div class="val">{r["latency"]["total_ms"]:.0f}<span style="font-size:1rem;color:#64748b">ms</span></div></div>',
                        unsafe_allow_html=True)

        # Stage flags
        st.markdown("**🛡️ FIREWALL STAGES**")
        f1 = r["stage1_entropy"].get("flagged", False)
        f2 = r["stage2_jsd"].get("flagged", False)
        f3 = r["stage3_nli"].get("flagged", False)

        badges = ""
        badges += f'<span class="stage-badge {"badge-fail" if f1 else "badge-pass"}">S1 Entropy {"⚠" if f1 else "✓"} {s1:.3f}</span>'
        badges += f'<span class="stage-badge {"badge-fail" if f2 else "badge-pass"}">S2 JSD {"⚠" if f2 else "✓"} {s2:.3f}</span>'
        badges += f'<span class="stage-badge {"badge-fail" if f3 else "badge-pass"}">S3 NLI {"⚠" if f3 else "✓"} {s3:.3f}</span>'
        st.markdown(badges, unsafe_allow_html=True)

        # Latency breakdown
        lat = r["latency"]
        with st.expander("⏱️ Latency Breakdown"):
            lat_data = {
                "Stage": ["Retrieval", "Answer Gen", "S1 Entropy", "S2 JSD", "S3 NLI", "TOTAL"],
                "Latency (ms)": [
                    st.session_state.last_result["ret_latency"],
                    lat["answer_ms"], lat["stage1_ms"],
                    lat["stage2_ms"], lat["stage3_ms"], lat["total_ms"],
                ],
            }
            st.dataframe(pd.DataFrame(lat_data), use_container_width=True, hide_index=True)

        # Retrieved chunks
        with st.expander(f"📄 Retrieved Chunks ({len(r['chunks'])})"):
            for i, chunk in enumerate(r["chunks"], 1):
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<div class="src">📄 {chunk["source"]} · Chunk {i}</div>'
                    f'{chunk["content"][:400]}{"..." if len(chunk["content"]) > 400 else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Sample outputs (entropy)
        samples = r["stage1_entropy"].get("samples", [])
        if samples:
            with st.expander(f"🎲 Entropy Samples ({len(samples)} stochastic outputs)"):
                for i, s in enumerate(samples, 1):
                    st.markdown(f"**Sample {i}:** {s}")
                    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("## EVALUATION DASHBOARD")

    from src.evaluation.logger import load_logs
    logs = load_logs()

    if not logs:
        st.info("No queries yet. Run some queries in the Query tab to populate the dashboard.")
    else:
        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # ── KPI Row ───────────────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                f'<div class="metric-card"><h4>Total Queries</h4>'
                f'<div class="val">{len(df)}</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            avg_risk = df["composite_risk_score"].mean()
            rc = "risk-low" if avg_risk < 0.3 else ("risk-medium" if avg_risk < 0.6 else "risk-high")
            st.markdown(
                f'<div class="metric-card"><h4>Avg Risk Score</h4>'
                f'<div class="val {rc}">{avg_risk:.2f}</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            flagged_pct = (df["stages_flagged"] > 0).mean() * 100
            st.markdown(
                f'<div class="metric-card"><h4>Flagged Rate</h4>'
                f'<div class="val">{flagged_pct:.0f}%</div></div>',
                unsafe_allow_html=True,
            )
        with col4:
            avg_faith = df["answer_faithfulness"].mean()
            st.markdown(
                f'<div class="metric-card"><h4>Avg Faithfulness</h4>'
                f'<div class="val">{avg_faith:.2f}</div></div>',
                unsafe_allow_html=True,
            )
        with col5:
            avg_lat = df["total_latency_ms"].mean()
            st.markdown(
                f'<div class="metric-card"><h4>Avg Latency</h4>'
                f'<div class="val">{avg_lat:.0f}<span style="font-size:1rem;color:#64748b">ms</span></div></div>',
                unsafe_allow_html=True,
            )

        # ── Charts Row ────────────────────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Risk Score Over Time**")
            fig_risk = px.line(
                df, x="timestamp", y="composite_risk_score",
                color_discrete_sequence=["#00d4ff"],
                markers=True,
            )
            fig_risk.add_hline(y=0.3, line_dash="dot", line_color="#22c55e", annotation_text="Low/Med")
            fig_risk.add_hline(y=0.6, line_dash="dot", line_color="#f59e0b", annotation_text="Med/High")
            fig_risk.update_layout(
                height=250, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", yaxis=dict(range=[0, 1], gridcolor="#2a3040"),
                xaxis=dict(gridcolor="#2a3040"),
                showlegend=False,
            )
            st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})

        with col_right:
            st.markdown("**RAGAS Metrics Over Time**")
            fig_ragas = go.Figure()
            for col, color in [
                ("context_precision", "#00d4ff"),
                ("answer_faithfulness", "#7c3aed"),
                ("answer_relevancy", "#22c55e"),
            ]:
                if col in df.columns:
                    fig_ragas.add_trace(go.Scatter(
                        x=df["timestamp"], y=df[col],
                        name=col.replace("_", " ").title(),
                        line=dict(color=color), mode="lines+markers",
                    ))
            fig_ragas.update_layout(
                height=250, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                legend=dict(
                    font=dict(size=10, family="JetBrains Mono"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                yaxis=dict(range=[0, 1], gridcolor="#2a3040"),
                xaxis=dict(gridcolor="#2a3040"),
            )
            st.plotly_chart(fig_ragas, use_container_width=True, config={"displayModeBar": False})

        # ── Stage scores avg ──────────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Average Stage Scores**")
            stage_avgs = {
                "S1 Entropy": df["entropy_score"].mean(),
                "S2 JSD": df["jsd_score"].mean(),
                "S3 NLI": df["nli_score"].mean(),
            }
            fig_stages = go.Figure(go.Bar(
                x=list(stage_avgs.keys()),
                y=list(stage_avgs.values()),
                marker_color=["#00d4ff", "#7c3aed", "#f59e0b"],
                text=[f"{v:.3f}" for v in stage_avgs.values()],
                textposition="outside",
                textfont={"family": "JetBrains Mono", "size": 11},
            ))
            fig_stages.update_layout(
                height=220, margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", showlegend=False,
                yaxis=dict(range=[0, 1], gridcolor="#2a3040"),
                xaxis=dict(gridcolor="#2a3040"),
            )
            st.plotly_chart(fig_stages, use_container_width=True, config={"displayModeBar": False})

        with col_b:
            st.markdown("**Risk Label Distribution**")
            risk_counts = df["risk_label"].value_counts()
            color_map = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
            fig_pie = go.Figure(go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker_colors=[color_map.get(l, "#64748b") for l in risk_counts.index],
                textfont={"family": "JetBrains Mono", "size": 11},
                hole=0.4,
            ))
            fig_pie.update_layout(
                height=220, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                legend=dict(font=dict(size=10, family="JetBrains Mono"), bgcolor="rgba(0,0,0,0)"),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        # ── Latency breakdown ─────────────────────────────────────────────────
        st.markdown("**Latency per Stage (ms)**")
        lat_cols = ["retrieval_latency_ms", "answer_latency_ms",
                    "stage1_latency_ms", "stage2_latency_ms", "stage3_latency_ms"]
        lat_labels = ["Retrieval", "Answer Gen", "S1 Entropy", "S2 JSD", "S3 NLI"]
        lat_avgs = [df[c].mean() for c in lat_cols if c in df.columns]

        fig_lat = go.Figure(go.Bar(
            x=lat_labels[:len(lat_avgs)], y=lat_avgs,
            marker_color="#00d4ff",
            text=[f"{v:.0f}ms" for v in lat_avgs],
            textposition="outside",
            textfont={"family": "JetBrains Mono", "size": 10},
        ))
        fig_lat.update_layout(
            height=200, margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", showlegend=False,
            yaxis=dict(gridcolor="#2a3040"),
            xaxis=dict(gridcolor="#2a3040"),
        )
        st.plotly_chart(fig_lat, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QUERY LOGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown("## QUERY LOGS")

    from src.evaluation.logger import load_logs, clear_logs as _clear_logs

    logs = load_logs()

    col_hdr, col_clr = st.columns([5, 1])
    with col_hdr:
        st.caption(f"{len(logs)} queries logged")
    with col_clr:
        if st.button("🗑️ Clear", use_container_width=True):
            _clear_logs()
            st.rerun()

    if not logs:
        st.info("No queries logged yet.")
    else:
        df_logs = pd.DataFrame(logs)
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"]).dt.strftime("%H:%M:%S")

        display_cols = [
            "timestamp", "query", "risk_label", "composite_risk_score",
            "context_precision", "answer_faithfulness",
            "entropy_score", "jsd_score", "nli_score",
            "total_latency_ms",
        ]
        display_cols = [c for c in display_cols if c in df_logs.columns]

        st.dataframe(
            df_logs[display_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "composite_risk_score": st.column_config.ProgressColumn(
                    "Risk Score", min_value=0, max_value=1, format="%.3f"
                ),
                "context_precision": st.column_config.ProgressColumn(
                    "Ctx Precision", min_value=0, max_value=1, format="%.3f"
                ),
                "answer_faithfulness": st.column_config.ProgressColumn(
                    "Faithfulness", min_value=0, max_value=1, format="%.3f"
                ),
            },
        )

        # Drill-down
        st.divider()
        st.markdown("**🔍 Log Detail**")
        idx = st.selectbox(
            "Select entry",
            range(len(logs)),
            format_func=lambda i: f"[{logs[i]['timestamp'][:19]}] {logs[i]['query'][:60]}...",
            label_visibility="collapsed",
        )
        if idx is not None:
            entry = logs[idx]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Query:** {entry['query']}")
                st.markdown(f"**Answer:** {entry['answer']}")
            with col_b:
                st.json({
                    "risk_label": entry["risk_label"],
                    "composite_risk_score": entry["composite_risk_score"],
                    "entropy_score": entry.get("entropy_score"),
                    "jsd_score": entry.get("jsd_score"),
                    "nli_score": entry.get("nli_score"),
                    "context_precision": entry.get("context_precision"),
                    "answer_faithfulness": entry.get("answer_faithfulness"),
                    "total_latency_ms": entry.get("total_latency_ms"),
                    "sources": entry.get("sources"),
                })
