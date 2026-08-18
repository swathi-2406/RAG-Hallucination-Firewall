"""
app.py — RAG Hallucination Firewall Dashboard
LLM: DeepSeek V4 Flash | Run: streamlit run app.py
"""

import sys, logging, time
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)

from config.settings import INDEX_DIR, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from config.settings import ENTROPY_THRESHOLD, JSD_THRESHOLD, NLI_THRESHOLD

st.set_page_config(page_title="RAG Hallucination Firewall", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
html,body,[data-testid="stAppViewContainer"]{background-color:#0d0f14!important;color:#e2e8f0;font-family:'Inter',sans-serif;}
[data-testid="stSidebar"]{background-color:#161a23!important;border-right:1px solid #2a3040;}
.stButton>button{background:linear-gradient(135deg,#00d4ff22,#7c3aed22);border:1px solid #00d4ff;color:#00d4ff;font-family:'JetBrains Mono',monospace;font-weight:700;border-radius:6px;padding:0.5rem 1.5rem;width:100%;}
.metric-card{background:#161a23;border:1px solid #2a3040;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:0.75rem;}
.metric-card h4{color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.4rem 0;font-family:'JetBrains Mono',monospace;}
.metric-card .val{font-size:2rem;font-weight:700;font-family:'JetBrains Mono',monospace;color:#e2e8f0;line-height:1;}
.answer-box{background:#1e2330;border:1px solid #2a3040;border-left:3px solid #00d4ff;border-radius:8px;padding:1.2rem 1.5rem;font-size:0.95rem;line-height:1.7;margin:1rem 0;}
.stage-badge{display:inline-block;padding:0.2rem 0.7rem;border-radius:999px;font-size:0.75rem;font-family:'JetBrains Mono',monospace;font-weight:700;margin-right:0.4rem;}
.badge-pass{background:#22c55e22;color:#22c55e;border:1px solid #22c55e;}
.badge-fail{background:#ef444422;color:#ef4444;border:1px solid #ef4444;}
.chunk-card{background:#161a23;border:1px solid #2a3040;border-radius:8px;padding:0.9rem 1.2rem;margin-bottom:0.6rem;font-size:0.82rem;font-family:'JetBrains Mono',monospace;color:#64748b;}
.chunk-card .src{color:#00d4ff;font-size:0.7rem;margin-bottom:0.3rem;}
h1,h2,h3{font-family:'JetBrains Mono',monospace;}
h1{color:#00d4ff;}
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input{background:#1e2330!important;border:1px solid #2a3040!important;color:#e2e8f0!important;font-family:'JetBrains Mono',monospace!important;border-radius:6px!important;}
label{color:#64748b!important;font-size:0.8rem!important;}
</style>
""", unsafe_allow_html=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    try:
        from src.retrieval.retriever import load_index
        return load_index()
    except FileNotFoundError:
        return None


def make_gauge(score, label):
    # composite risk is now normalized against each stage's own calibrated
    # threshold (1.0 = at threshold), so it can exceed 1.0; cap the gauge
    # display at 100% while keeping the color cutoffs at 0.5/1.0
    color = "#22c55e" if score < 0.35 else ("#f59e0b" if score < 0.80 else "#ef4444")
    score = min(score, 1.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score*100,1),
        number={"suffix":"%","font":{"size":28,"color":color,"family":"JetBrains Mono"}},
        gauge={"axis":{"range":[0,100],"tickfont":{"color":"#64748b","size":10}},
               "bar":{"color":color,"thickness":0.25},"bgcolor":"#1e2330","bordercolor":"#2a3040",
               "steps":[{"range":[0,30],"color":"rgba(34,197,94,0.09)"},
                        {"range":[30,60],"color":"rgba(245,158,11,0.09)"},
                        {"range":[60,100],"color":"rgba(239,68,68,0.09)"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":round(score*100,1)}},
        title={"text":label,"font":{"size":11,"color":"#64748b","family":"JetBrains Mono"}},
    ))
    fig.update_layout(height=200,margin=dict(l=20,r=20,t=30,b=10),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0")
    return fig


def make_stage_bars(s1,s2,s3):
    stages=["Stage 1<br>Entropy","Stage 2<br>JSD","Stage 3<br>NLI"]
    scores=[s1,s2,s3]; thresholds=[ENTROPY_THRESHOLD,JSD_THRESHOLD,NLI_THRESHOLD]
    colors=["#22c55e" if s<t else "#ef4444" for s,t in zip(scores,thresholds)]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=stages,y=scores,marker_color=colors,
                         text=[f"{s:.3f}" for s in scores],textposition="outside",
                         textfont={"family":"JetBrains Mono","size":11,"color":"#e2e8f0"}))
    for i,(t,_) in enumerate(zip(thresholds,stages)):
        fig.add_shape(type="line",x0=i-0.4,x1=i+0.4,y0=t,y1=t,
                      line=dict(color="#f59e0b",width=2,dash="dot"))
    fig.update_layout(height=220,margin=dict(l=10,r=10,t=20,b=10),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e2e8f0",yaxis=dict(range=[0,1],gridcolor="#2a3040"),
                      xaxis=dict(tickfont={"family":"JetBrains Mono","size":10}),
                      showlegend=False,bargap=0.4)
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:JetBrains Mono;font-size:1.4rem;font-weight:700;color:#00d4ff;">🔥 RAG Firewall</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.8rem;font-family:JetBrains Mono;">Hallucination Detection System</div>', unsafe_allow_html=True)
    st.divider()

    # LLM status
    if DEEPSEEK_API_KEY:
        st.success(f"✅ DeepSeek V4 Flash active")
    else:
        st.error("⚠️ DEEPSEEK_API_KEY not set in .env")
        st.caption("Sign up free at platform.deepseek.com")

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
    top_k   = st.slider("Chunks to retrieve (k)", 1, 10, 5)
    mmr_div = st.slider("MMR Diversity", 0.0, 1.0, 0.3, 0.05)
    st.divider()
    st.markdown("**📖 About**")
    st.caption("Stage 1: Semantic Entropy\nStage 2: Jensen-Shannon Divergence\nStage 3: DeBERTa NLI\n\nLLM: DeepSeek V4 Flash")


# ── TABS ──────────────────────────────────────────────────────────────────────
tab_query, tab_dashboard, tab_logs = st.tabs(["🔍 Query", "📊 Dashboard", "📋 Query Logs"])


# ══════════════════════════════════════════════════════
# TAB 1: QUERY
# ══════════════════════════════════════════════════════
with tab_query:
    st.markdown("## QUERY INTERFACE")
    vs = load_vectorstore()

    col_q, col_btn = st.columns([5,1])
    with col_q:
        query = st.text_area("Question", placeholder="e.g. What is retrieval-augmented generation?",
                             height=80, label_visibility="collapsed")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡ RUN", use_container_width=True)

    if run_btn and query.strip():
        if not vs:
            st.error("FAISS index not loaded. Run `python scripts/ingest_docs.py` first.")
        elif not DEEPSEEK_API_KEY:
            st.error("Add DEEPSEEK_API_KEY to .env")
        else:
            with st.spinner("Running pipeline..."):
                from src.retrieval.retriever import retrieve, get_context_string
                from src.hallucination.firewall import run_firewall
                from src.evaluation.metrics import compute_all_metrics
                from src.evaluation.logger import log_query

                chunks, ret_latency = retrieve(query, vs, top_k=top_k, diversity=mmr_div)
                result = run_firewall(query, chunks, run_entropy=run_entropy, run_jsd=run_jsd, run_nli=run_nli)
                metrics = compute_all_metrics(query, result["answer"], [c.page_content for c in chunks])
                log_query(query, result["answer"], result, metrics, ret_latency)
                st.session_state.last_result = {"result":result,"metrics":metrics,"ret_latency":ret_latency}

    if st.session_state.last_result:
        r = st.session_state.last_result["result"]
        m = st.session_state.last_result["metrics"]
        risk = r["composite_risk_score"]
        label = r["risk_label"]
        backend = r.get("answer_backend","unknown")

        st.divider()
        st.markdown(f"**📝 ANSWER** <span style='font-size:0.75rem;color:#64748b;font-family:JetBrains Mono;'>via {backend}</span>", unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{r["answer"]}</div>', unsafe_allow_html=True)

        col_gauge, col_stages, col_meta = st.columns([2,3,2])
        with col_gauge:
            st.plotly_chart(make_gauge(risk, f"{label} RISK"), use_container_width=True, config={"displayModeBar":False})
        with col_stages:
            s1=r["stage1_entropy"].get("score",0); s2=r["stage2_jsd"].get("score",0); s3=r["stage3_nli"].get("score",0)
            st.plotly_chart(make_stage_bars(s1,s2,s3), use_container_width=True, config={"displayModeBar":False})
        with col_meta:
            st.markdown(f'<div class="metric-card"><h4>Context Precision</h4><div class="val">{m["context_precision"]:.2f}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h4>Answer Faithfulness</h4><div class="val">{m["answer_faithfulness"]:.2f}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h4>Total Latency</h4><div class="val">{r["latency"]["total_ms"]:.0f}<span style="font-size:1rem;color:#64748b">ms</span></div></div>', unsafe_allow_html=True)

        st.markdown("**🛡️ FIREWALL STAGES**")
        f1=r["stage1_entropy"].get("flagged",False); f2=r["stage2_jsd"].get("flagged",False); f3=r["stage3_nli"].get("flagged",False)
        badges = (f'<span class="stage-badge {"badge-fail" if f1 else "badge-pass"}">S1 Entropy {"⚠" if f1 else "✓"} {s1:.3f}</span>'
                  f'<span class="stage-badge {"badge-fail" if f2 else "badge-pass"}">S2 JSD {"⚠" if f2 else "✓"} {s2:.3f}</span>'
                  f'<span class="stage-badge {"badge-fail" if f3 else "badge-pass"}">S3 NLI {"⚠" if f3 else "✓"} {s3:.3f}</span>')
        st.markdown(badges, unsafe_allow_html=True)

        lat=r["latency"]
        with st.expander("⏱️ Latency Breakdown"):
            st.dataframe(pd.DataFrame({
                "Stage":["Retrieval","Answer Gen","S1 Entropy","S2 JSD","S3 NLI","TOTAL"],
                "Latency (ms)":[st.session_state.last_result["ret_latency"],
                                lat["answer_ms"],lat["stage1_ms"],lat["stage2_ms"],lat["stage3_ms"],lat["total_ms"]]
            }), use_container_width=True, hide_index=True)

        with st.expander(f"📄 Retrieved Chunks ({len(r['chunks'])})"):
            for i,chunk in enumerate(r["chunks"],1):
                st.markdown(f'<div class="chunk-card"><div class="src">📄 {chunk["source"]} · Chunk {i}</div>{chunk["content"][:400]}{"..." if len(chunk["content"])>400 else ""}</div>', unsafe_allow_html=True)

        samples=r["stage1_entropy"].get("samples",[])
        if samples:
            with st.expander(f"🎲 Entropy Samples ({len(samples)} outputs)"):
                for i,s in enumerate(samples,1):
                    st.markdown(f"**Sample {i}:** {s}"); st.divider()


# ══════════════════════════════════════════════════════
# TAB 2: DASHBOARD
# ══════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("## EVALUATION DASHBOARD")
    from src.evaluation.logger import load_logs
    logs = load_logs()

    if not logs:
        st.info("No queries yet. Run some queries to populate the dashboard.")
    else:
        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        col1,col2,col3,col4,col5 = st.columns(5)
        for col,title,val,is_risk in [
            (col1,"Total Queries",str(len(df)),False),
            (col2,"Avg Risk Score",f"{df['composite_risk_score'].mean():.2f}",True),
            (col3,"Flagged Rate",f"{(df['stages_flagged']>0).mean()*100:.0f}%",False),
            (col4,"Avg Faithfulness",f"{df['answer_faithfulness'].mean():.2f}",False),
            (col5,"Avg Latency",f"{df['total_latency_ms'].mean():.0f}ms",False),
        ]:
            with col:
                color = "#22c55e" if is_risk and df["composite_risk_score"].mean()<0.35 else "#f59e0b" if is_risk else "#e2e8f0"
                st.markdown(f'<div class="metric-card"><h4>{title}</h4><div class="val" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Risk Score Over Time**")
            fig=px.line(df,x="timestamp",y="composite_risk_score",color_discrete_sequence=["#00d4ff"],markers=True)
            fig.add_hline(y=0.35,line_dash="dot",line_color="#22c55e")
            fig.add_hline(y=0.80,line_dash="dot",line_color="#f59e0b")
            fig.update_layout(height=250,margin=dict(l=10,r=10,t=10,b=10),
                              paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e2e8f0",yaxis=dict(range=[0,1],gridcolor="#2a3040"),
                              xaxis=dict(gridcolor="#2a3040"),showlegend=False)
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        with col_right:
            st.markdown("**RAGAS Metrics Over Time**")
            fig2=go.Figure()
            for col_name,color in [("context_precision","#00d4ff"),("answer_faithfulness","#7c3aed"),("answer_relevancy","#22c55e")]:
                if col_name in df.columns:
                    fig2.add_trace(go.Scatter(x=df["timestamp"],y=df[col_name],
                                              name=col_name.replace("_"," ").title(),
                                              line=dict(color=color),mode="lines+markers"))
            fig2.update_layout(height=250,margin=dict(l=10,r=10,t=10,b=10),
                               paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0",legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"),
                               yaxis=dict(range=[0,1],gridcolor="#2a3040"),xaxis=dict(gridcolor="#2a3040"))
            st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

        col_a,col_b=st.columns(2)
        with col_a:
            st.markdown("**Average Stage Scores**")
            fig3=go.Figure(go.Bar(
                x=["S1 Entropy","S2 JSD","S3 NLI"],
                y=[df["entropy_score"].mean(),df["jsd_score"].mean(),df["nli_score"].mean()],
                marker_color=["#00d4ff","#7c3aed","#f59e0b"],
                text=[f"{v:.3f}" for v in [df["entropy_score"].mean(),df["jsd_score"].mean(),df["nli_score"].mean()]],
                textposition="outside",textfont={"family":"JetBrains Mono","size":11}))
            fig3.update_layout(height=220,margin=dict(l=10,r=10,t=20,b=10),
                               paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0",showlegend=False,
                               yaxis=dict(range=[0,1],gridcolor="#2a3040"))
            st.plotly_chart(fig3,use_container_width=True,config={"displayModeBar":False})

        with col_b:
            st.markdown("**Risk Label Distribution**")
            rc=df["risk_label"].value_counts()
            color_map={"LOW":"#22c55e","MEDIUM":"#f59e0b","HIGH":"#ef4444"}
            fig4=go.Figure(go.Pie(labels=rc.index,values=rc.values,
                                   marker_colors=[color_map.get(l, "#64748b") for l in rc.index],
                                   hole=0.4,textfont={"family":"JetBrains Mono","size":11}))
            fig4.update_layout(height=220,margin=dict(l=10,r=10,t=10,b=10),
                               paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0",legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig4,use_container_width=True,config={"displayModeBar":False})


# ══════════════════════════════════════════════════════
# TAB 3: LOGS
# ══════════════════════════════════════════════════════
with tab_logs:
    st.markdown("## QUERY LOGS")
    from src.evaluation.logger import load_logs, clear_logs as _clear
    logs = load_logs()

    col_hdr, col_clr = st.columns([5,1])
    with col_hdr: st.caption(f"{len(logs)} queries logged")
    with col_clr:
        if st.button("🗑️ Clear", use_container_width=True):
            _clear(); st.rerun()

    if not logs:
        st.info("No queries logged yet.")
    else:
        df_logs=pd.DataFrame(logs)
        df_logs["timestamp"]=pd.to_datetime(df_logs["timestamp"]).dt.strftime("%H:%M:%S")
        display_cols=["timestamp","query","answer_backend","risk_label","composite_risk_score",
                      "context_precision","answer_faithfulness","entropy_score","jsd_score","nli_score","total_latency_ms"]
        display_cols=[c for c in display_cols if c in df_logs.columns]
        st.dataframe(df_logs[display_cols].sort_values("timestamp",ascending=False),
                     use_container_width=True, hide_index=True,
                     column_config={
                         "composite_risk_score":st.column_config.ProgressColumn("Risk",min_value=0,max_value=1,format="%.3f"),
                         "context_precision":st.column_config.ProgressColumn("Precision",min_value=0,max_value=1,format="%.3f"),
                         "answer_faithfulness":st.column_config.ProgressColumn("Faithful",min_value=0,max_value=1,format="%.3f"),
                     })

        st.divider()
        idx=st.selectbox("Log detail",range(len(logs)),
                         format_func=lambda i:f"[{logs[i]['timestamp'][:19]}] {logs[i]['query'][:60]}...",
                         label_visibility="collapsed")
        if idx is not None:
            entry=logs[idx]
            col_a,col_b=st.columns(2)
            with col_a:
                st.markdown(f"**Query:** {entry['query']}")
                st.markdown(f"**Answer:** {entry['answer']}")
                st.caption(f"LLM backend: {entry.get('answer_backend','unknown')}")
            with col_b:
                st.json({k:entry[k] for k in ["risk_label","composite_risk_score","entropy_score",
                         "jsd_score","nli_score","context_precision","answer_faithfulness","total_latency_ms","sources"] if k in entry})
