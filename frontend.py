# -*- coding: utf-8 -*-
# frontend.py — Streamlit UI for rag_pipeline.py
# CS6120 project build: header line, manifest-hash rebuild, chunks-per-page control,
# selection-driven Sentiment, model-select Summarization, ROUGE Metrics.
# 13 AUG 2025 - Rajorshi Sarkar / Priyan Rupesh Mehta

import os, json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import sys
sys.path.append('.')

# Pipeline bits
from rag_pipeline import (
    ingest_pdfs, DatasetManager, EmbeddingSystem,
    QASystem, SentimentAnalyzer, Summarizer,
    setup_kaggle_credentials,
    clean_model_output
)

# ---- extra NLP utils for model-switchable summarization ----
from transformers import pipeline, PegasusForConditionalGeneration, PegasusTokenizer
from rouge_score import rouge_scorer

# Corrected model dictionary: The label for the 'samsum' model is changed
# from "RoBERTa (summarization)" to "BART (samsum)" to be more accurate.
SUMM_MODELS = {
    "DistilBART (fast, good)": "sshleifer/distilbart-cnn-12-6",
    "BART-Large (best, slower)": "facebook/bart-large-cnn",
    "T5-small (fastest)": "t5-small",
    "BART (samsum)": "philschmid/bart-large-cnn-samsum", # Relabeled from "RoBERTa (summarization)"
    "Pegasus (best, slower)": "google/pegasus-cnn_dailymail"
}

@st.cache_resource
def get_sum_pipe(model_id: str):
    """
    Caches the summarization pipeline to avoid reloading models on each rerun.
    Handles special initialization for Pegasus which uses a different pipeline setup.
    """
    if "pegasus" in model_id.lower():
        return {"tokenizer": PegasusTokenizer.from_pretrained(model_id),
                "model": PegasusForConditionalGeneration.from_pretrained(model_id)}
    return pipeline("summarization", model=model_id)

def run_summarize(pipe, text: str, model_label: str, max_len=150, min_len=30) -> str:
    """
    Generates a summary from a given text using the selected model.
    Handles specific generation parameters for Pegasus and BART/RoBERTa models.
    """
    if not text:
        return ""
    
    if "Pegasus" in model_label:
        default_params = {
            "max_length": max_len,
            "min_length": min_len,
            "length_penalty": 2.0,
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
            "do_sample": False
        }
        inputs = pipe["tokenizer"](text[:1024], return_tensors="pt", truncation=True)
        summary_ids = pipe["model"].generate(inputs["input_ids"], **default_params)
        summary_text = pipe["tokenizer"].decode(summary_ids[0], skip_special_tokens=True)
        return summary_text
    else:
        generate_kwargs = {"max_length": max_len, "max_new_tokens": None}
        if "bart" in pipe.model.name_or_path.lower() or "roberta" in pipe.model.name_or_path.lower():
            generate_kwargs["min_length"] = min_len

        out = pipe(text[:1024], **generate_kwargs)
        summary_text = out[0].get("summary_text", out[0].get("generated_text", ""))
        return clean_model_output(summary_text)

def run_summarize_long(pipe, text: str, model_label: str, window_chars: int = 1000) -> str:
    """
    Summarizes long text by chunking, summarizing each chunk, and then summarizing the summaries.
    """
    if not text:
        return ""
    pieces = [text[i:i+window_chars] for i in range(0, len(text), window_chars)]
    partials = [run_summarize(pipe, p, model_label) for p in pieces]
    return run_summarize(pipe, " ".join(partials), model_label)

_ROUGE = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"])
def rouge_scores(ref: str, hyp: str) -> dict:
    """
    Calculates ROUGE scores (rouge1, rouge2, rougeL) for a summary against a reference text.
    """
    s = _ROUGE.score(ref[:500], hyp or "")
    return {k: v.fmeasure for k, v in s.items()}

# ---------- preview helper to avoid backslashes in f-strings ----------
def one_line_preview(text: str, n: int = 80) -> str:
    """Creates a single-line preview of a text string."""
    return " ".join((text or "")[:n].splitlines())

# ---------------- Session State Management ----------------
# Initializes session state variables to store data across Streamlit reruns.
if "dm" not in st.session_state: st.session_state.dm = DatasetManager()
if "docs_preview" not in st.session_state: st.session_state.docs_preview = []
if "ready" not in st.session_state: st.session_state.ready = False
if "es" not in st.session_state: st.session_state.es = None
if "qa" not in st.session_state: st.session_state.qa = None
if "sent" not in st.session_state: st.session_state.sent = None  
if "summ" not in st.session_state: st.session_state.summ = None  
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "qa_results" not in st.session_state: st.session_state.qa_results = []
if "last_chunks" not in st.session_state: st.session_state.last_chunks = []
if "metrics_log" not in st.session_state: st.session_state.metrics_log = {"rouge_runs": []}

# ----------- cached lazy loaders -----------
@st.cache_resource
def _load_qa(_es):
    """Loads and caches the QA models."""
    qa = QASystem(_es); qa.load_models()
    return qa

@st.cache_resource
def _load_sent():
    """Loads and caches the sentiment analysis models."""
    return SentimentAnalyzer()

@st.cache_resource
def _load_summ():
    """Loads and caches the summarization pipeline."""
    return Summarizer()

def manifest_df(docs: list) -> pd.DataFrame:
    """Converts a list of document records into a Pandas DataFrame for display."""
    rows = []
    for d in docs:
        m = d.get("metadata", {})
        rows.append({
            "title": m.get("title",""),
            "page": m.get("page", ""),
            "chars": len(d.get("text","")),
            "path": m.get("path","")
        })
    return pd.DataFrame(rows)

# ------- plotting helpers -------
def sentiment_bar(labels: list):
    """Generates a Plotly bar chart for sentiment distribution."""
    if not labels:
        return go.Figure()
    vc = pd.Series(labels).value_counts().reindex(["positive","neutral","negative"]).fillna(0)
    fig = go.Figure([go.Bar(x=vc.index, y=vc.values)])
    fig.update_layout(title="Sentiment Distribution", height=300, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def rouge_bar(scores: dict):
    """Generates a Plotly bar chart for ROUGE scores."""
    fig = go.Figure([go.Bar(x=list(scores.keys()), y=list(scores.values()))])
    fig.update_layout(title="ROUGE Scores", yaxis_range=[0,1], height=300, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ---------- index/embeddings builder with progress + reuse ----------
def build_everything(docs: list, *, force_rebuild: bool, chunks_per_page: int, batch_size: int):
    """
    Orchestrates the entire process of document ingestion, chunking, embedding,
    and FAISS index creation, with progress updates. It includes logic to reuse
    an existing index if the document set hasn't changed.
    """
    prog = st.progress(0); status = st.empty()

    from hashlib import sha256
    manifest_hash = sha256(json.dumps([
        {"path": d.get("metadata",{}).get("path",""),
         "page": d.get("metadata",{}).get("page",0),
         "title": d.get("metadata",{}).get("title",""),
         "chars": len(d.get("text",""))}
        for d in docs
    ], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    # Reuse existing index only if manifest matches and not force_rebuild
    if Path("faiss_index.bin").exists() and Path("chunks.pkl").exists() and Path("data/manifest.hash").exists() and not force_rebuild:
        try:
            old_hash = Path("data/manifest.hash").read_text(encoding="utf-8")
            if old_hash == manifest_hash:
                status.text("📂 Reusing existing FAISS index (no changes detected)…")
                es = EmbeddingSystem(); es.load()
                st.session_state.es = es
                st.session_state.ready = True
                prog.progress(100)
                status.text("✅ Index ready (reused)")
                return
            else:
                status.text("📄 Changes detected in documents — rebuilding index…")
        except Exception:
            pass  # fall back to rebuild

    status.text("Step 1/3: Initializing embedding model… (first run may download)")
    st.session_state.es = EmbeddingSystem()
    prog.progress(10)
    
    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/all_documents.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    (Path("data")/"manifest.hash").write_text(manifest_hash, encoding="utf-8")

    status.text("Step 2/3: Creating chunks…")
    chunks = st.session_state.es.create_chunks(docs, max_chars_per_page=2000, chunks_per_page=chunks_per_page)
    prog.progress(40)

    status.text(f"Step 3/3: Embedding {len(chunks)} chunks & building FAISS…")
    st.session_state.es.generate_embeddings(batch_size=batch_size)
    st.session_state.es.create_faiss_index()
    st.session_state.es.save()
    prog.progress(100)

    st.session_state.qa = None
    st.session_state.sent = None
    st.session_state.summ = None
    st.session_state.ready = True
    status.text("✅ Index ready (built)")

# ---------------- Sidebar: Build Corpus ----------------
st.sidebar.title("🎛️ Build Corpus")

def clear_cache_and_session():
    """Clears all session state and Streamlit resource caches."""
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()

if st.sidebar.button("Remove/Clear Cache", type="secondary"):
    clear_cache_and_session()
    st.success("✅ Cache and session state cleared.")

st.sidebar.markdown("---")

st.sidebar.subheader("Upload PDFs")
pdf_files = st.sidebar.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_files")

with st.sidebar.expander("Kaggle (optional)"):
    u = st.text_input("Username", key="kg_user")
    k = st.text_input("API Key", type="password", key="kg_key")
    if st.button("Set Kaggle Credentials", key="btn_set_kaggle"):
        if u and k:
            setup_kaggle_credentials(u, k)
            st.success("Kaggle credentials saved.")
        else:
            st.warning("Enter username and key.")

st.sidebar.markdown("---")
only_pdfs = st.sidebar.checkbox("Only index uploaded PDFs (ignore datasets)", value=True, key="only_pdfs")

use_gov = st.sidebar.checkbox("GovReport", value=False, key="chk_gov")
use_arxiv = st.sidebar.checkbox("arXiv (needs Kaggle)", value=False, key="chk_arxiv")
use_cord = st.sidebar.checkbox("CORD-19 (needs Kaggle)", value=False, key="chk_cord")

st.sidebar.subheader("GovReport Filters")
gov_kw = st.sidebar.text_input("Keywords (comma-separated)", "", key="gov_kw")
gov_max = st.sidebar.number_input("Max docs", min_value=0, max_value=10000, value=100, step=50, key="gov_max")

st.sidebar.subheader("arXiv Filters")
arxiv_kw = st.sidebar.text_input("Keywords (comma-separated)", "", key="arxiv_kw")
arxiv_cats = st.sidebar.text_input("Categories", "cs.CL,cs.AI,cs.LG", key="arxiv_cats")
c1, c2 = st.sidebar.columns(2)
with c1: arxiv_from = st.date_input("From", key="arxiv_from")
with c2: arxiv_to = st.date_input("To", key="arxiv_to")
arxiv_max = st.sidebar.number_input("Max docs", min_value=0, max_value=100000, value=500, step=100, key="arxiv_max")

st.sidebar.subheader("CORD-19 Filters")
cord_max = st.sidebar.number_input("Max docs", min_value=0, max_value=100000, value=500, step=100, key="cord_max")

force_rebuild = st.sidebar.checkbox("Force rebuild (ignore cached index)", value=False, key="force_rebuild")
chunks_per_page = st.sidebar.slider("Chunks per page (chunk granularity)", 1, 6, 1, key="chunks_per_page")
batch_size = st.sidebar.selectbox("Embedding Batch Size", options=[2, 3, 4, 8], index=0, key="batch_size")


if st.sidebar.button("🔎 Preview Selection (no indexing)", key="btn_preview"):
    """
    Button to show a preview of the documents that will be ingested, without
    actually building the index.
    """
    st.session_state.docs_preview = []
    docs = []

    if pdf_files:
        save_dir = Path("data/uploads"); save_dir.mkdir(parents=True, exist_ok=True)
        for uf in pdf_files:
            out = save_dir / uf.name
            with open(out, "wb") as w: w.write(uf.read())
        docs += ingest_pdfs(save_dir)

    if not only_pdfs:
        if use_gov:
            docs += st.session_state.dm.download_govreport(limit=gov_max, keywords=gov_kw)
        if use_arxiv:
            docs += st.session_state.dm.download_arxiv(
                keywords=arxiv_kw, categories=arxiv_cats,
                date_from=str(arxiv_from) if arxiv_from else None,
                date_to=str(arxiv_to) if arxiv_to else None,
                limit=arxiv_max
            )
        if use_cord:
            docs += st.session_state.dm.download_cord19(limit=cord_max)

    st.session_state.docs_preview = docs
    if docs:
        st.success(f"Prepared preview for {len(docs)} pages.")
        st.dataframe(manifest_df(docs), use_container_width=True)
    else:
        st.info("No documents selected or found.")


if st.sidebar.button("🧱 Ingest & Build Index", key="btn_build"):
    """
    Button to start the full ingestion and indexing process.
    It collects documents based on user selections and calls the main build function.
    """
    docs_to_ingest = st.session_state.docs_preview
    if not docs_to_ingest:
        docs_to_ingest = []
        if pdf_files:
            save_dir = Path("data/uploads"); save_dir.mkdir(parents=True, exist_ok=True)
            for uf in pdf_files:
                out = save_dir / uf.name
                with open(out, "wb") as w: w.write(uf.read())
            docs_to_ingest += ingest_pdfs(save_dir)

        if not only_pdfs:
            if use_gov:
                docs_to_ingest += st.session_state.dm.download_govreport(limit=gov_max, keywords=gov_kw)
            if use_arxiv:
                docs_to_ingest += st.session_state.dm.download_arxiv(
                    keywords=arxiv_kw, categories=arxiv_cats,
                    date_from=str(arxiv_from) if arxiv_from else None,
                    date_to=str(arxiv_to) if arxiv_to else None,
                    limit=arxiv_max
                )
            if use_cord:
                docs_to_ingest += st.session_state.dm.download_cord19(limit=cord_max)

    if not docs_to_ingest:
        st.warning("No documents selected or found for ingestion.")
    else:
        build_everything(docs_to_ingest, force_rebuild=force_rebuild, chunks_per_page=chunks_per_page, batch_size=batch_size)
        st.sidebar.success("Index ready.")

st.sidebar.markdown("---")
chunks_k = st.sidebar.slider("Top-k retrieved chunks (per query)", 1, 10, 5, key="chunks_k")
models_selected = st.sidebar.multiselect("Q&A Models (display only)", ["distilbert", "roberta", "bart_qa", "t5"],
                                         default=["distilbert", "roberta", "bart_qa", "t5"], key="models_select")

with st.sidebar.expander("Advanced Retrieval Options", expanded=False):
    use_reranker = st.checkbox("Enable Reranker", value=False, key="use_reranker")
    if use_reranker:
        rerank_initial_k = st.slider("Reranker initial pool (k)", 10, 50, 25, key="rerank_initial_k")
    else:
        rerank_initial_k = chunks_k
        
    use_mmr = st.checkbox("Enable MMR Diversity", value=False, key="use_mmr")
    if use_mmr:
        mmr_lambda = st.slider("MMR Lambda (rel. vs div.)", 0.0, 1.0, 0.5, step=0.1, key="mmr_lambda")
    
    use_kmeans = st.checkbox("Enable K-means Clustering", value=False, key="use_kmeans")
    if use_kmeans:
        kmeans_k = st.slider("K-means Clusters (k)", 2, chunks_k, 3, key="kmeans_k")


# ---------------- Header ----------------
cH1, cH2, cH3 = st.columns([2,4,2])
with cH2:
    st.title("🤖 RAG System Dashboard")
    st.markdown("**Retrieval-Augmented Generation with Q&A, Sentiment & Summarization**")
    st.markdown("_CS6120 Natural Language Processing Project  Rajorshi Sarkar , Priya Rupesh Mehta_")
    st.markdown("_Northeastern University 2025 Summer_")
st.markdown("---")

if not st.session_state.ready:
    st.info("👉 Build your corpus in the left sidebar (select sources, preview, then **Ingest & Build Index**).")

tabs = st.tabs(["💬 Q&A Chat","🔍 Embedding Search","🎭 Sentiment","📝 Summarization","📈 Metrics"])

# ---------------- Q&A Tab ----------------
with tabs[0]:
    st.header("💬 Interactive Q&A")
    if st.session_state.ready and st.session_state.qa is None:
        with st.spinner("Loading Q&A models (first time may download)…"):
            st.session_state.qa = _load_qa(st.session_state.es)

    qcol1,qcol2 = st.columns([3,1])
    with qcol1: q = st.text_input("Ask a question:", placeholder="e.g., What is the paper about?", key="qa_question")
    with qcol2: ask = st.button("🚀 Ask", use_container_width=True, key="qa_ask")

    if ask:
        if not st.session_state.ready:
            st.warning("Build the index first.")
        elif not q:
            st.warning("Type a question.")
        else:
            with st.spinner("Answering..."):
                # Call the QA system with user-defined retrieval parameters
                qa_out = st.session_state.qa.answer(
                    question=q, 
                    model_name="all", 
                    k=chunks_k, 
                    rerank_enabled=use_reranker, 
                    rerank_initial_k=rerank_initial_k, 
                    mmr_enabled=use_mmr, 
                    mmr_lambda=mmr_lambda, 
                    kmeans_enabled=use_kmeans, 
                    kmeans_k=kmeans_k
                )
                
                # Store results in session state for other tabs
                st.session_state.last_chunks = qa_out['retrieved']
                entry = {"timestamp": datetime.now().isoformat(), "query": q, "results": qa_out, "chunks": qa_out['retrieved']}
                st.session_state.chat_history.append(entry)
                st.session_state.qa_results.append(entry)

    if st.session_state.qa_results:
        latest = st.session_state.qa_results[-1]
        st.subheader("📋 Answers")
        # Display answers from selected models
        for m, a in latest["results"]["answers"].items():
            if m in models_selected:
                label = m.upper()
                if m == "bart_qa":
                    label = "BART QA"
                with st.expander(label, expanded=True):
                    st.write(a)
        with st.expander("📚 Retrieved Context"):
            # Display retrieved chunks for context
            for i, r in enumerate(latest["results"]["retrieved"], start=1):
                md = r["chunk"]["metadata"]; sc = r.get("score", None)
                st.markdown(f"**Chunk {i}** — {md.get('title','')} (p.{md.get('page','?')}){(' • score: %.3f'%sc) if sc is not None else ''}")
                st.write(r["chunk"]["text"][:500] + ("..." if len(r["chunk"]["text"])>500 else ""))

# ---------------- Embedding Search Tab ----------------
with tabs[1]:
    st.header("🔍 Embedding Search")
    if not st.session_state.ready:
        st.info("Build the index first.")
    else:
        q = st.text_input("Search vector space:", key="embed_query")
        if st.button("Search", key="btn_embed"):
            # Perform a simple semantic search and display results
            res = st.session_state.es.search(q, k=chunks_k)
            st.session_state.last_chunks = res
            for i,r in enumerate(res, start=1):
                st.write(f"**#{i}** score={r.get('score', float('nan')):.3f} — {r['chunk']['metadata'].get('title','')} (p.{r['chunk']['metadata'].get('page','?')})")
                st.write(r["chunk"]["text"][:400] + "…")

# ---------------- Sentiment Tab ----------------
with tabs[2]:
    st.header("🎭 Sentiment")
    if st.session_state.ready and st.session_state.sent is None:
        with st.spinner("Loading sentiment models…"):
            st.session_state.sent = _load_sent()

    mode = st.radio("Source", ["Last Retrieved", "Pick Document & Chunks", "Custom Text"], index=0, key="sent_mode")

    def _labels_for_res(res):
        """Helper to create display labels for a list of search results."""
        labels = []
        for i, r in enumerate(res):
            md = r["chunk"]["metadata"]
            preview = one_line_preview(r["chunk"]["text"], 80)
            labels.append(f"{i+1}. {md.get('title','')} (p.{md.get('page','?')}) — {preview}…")
        return labels

    if mode == "Last Retrieved":
        res = st.session_state.get("last_chunks", [])
        if not res:
            st.info("No recent retrieval. Ask a question in Q&A, or use 'Pick Document & Chunks'.")
        else:
            labels = _labels_for_res(res)
            sel = st.multiselect("Select chunks to analyze:", options=list(range(len(res))),
                                 format_func=lambda i: labels[i],
                                 default=list(range(len(res))), key="sent_last_sel")
            chosen = [res[i] for i in sel]
            if chosen and st.button("Analyze selected", key="btn_sent_sel"):
                results = [st.session_state.sent.analyze(r["chunk"]["text"]) for r in chosen]
                
                # Display sentiment analysis results from DistilBERT
                st.subheader("DistilBERT Analysis")
                distilbert_sentiments = [r["distilbert"]["sentiment"] for r in results]
                st.plotly_chart(sentiment_bar(distilbert_sentiments), use_container_width=True)
                a,b,c = st.columns(3)
                with a: st.metric("Positive", f"{distilbert_sentiments.count('positive')}/{len(chosen)}")
                with b: st.metric("Neutral", f"{distilbert_sentiments.count('neutral')}/{len(chosen)}")
                with c: st.metric("Negative", f"{distilbert_sentiments.count('negative')}/{len(chosen)}")
                
                # Display sentiment analysis results from RoBERTa
                st.subheader("RoBERTa Analysis")
                roberta_sentiments = [r["roberta"]["sentiment"] for r in results]
                st.plotly_chart(sentiment_bar(roberta_sentiments), use_container_width=True)
                d,e,f = st.columns(3)
                with d: st.metric("Positive", f"{roberta_sentiments.count('positive')}/{len(chosen)}")
                with e: st.metric("Neutral", f"{roberta_sentiments.count('neutral')}/{len(chosen)}")
                with f: st.metric("Negative", f"{roberta_sentiments.count('negative')}/{len(chosen)}")
                
                with st.expander("Selected chunk texts"):
                    for i, r in enumerate(chosen, 1):
                        md = r["chunk"]["metadata"]
                        st.write(f"**Chunk {i}** — {md.get('title','')} (p.{md.get('page','?')})")
                        st.write(r["chunk"]["text"])

    elif mode == "Pick Document & Chunks":
        # Allows sentiment analysis on specific chunks from a selected document
        if not st.session_state.ready or not st.session_state.es:
            st.info("Build the index first.")
        else:
            docs = {}
            for c in st.session_state.es.chunks:
                md = c.get("metadata", {})
                key = md.get("path") or md.get("title")
                docs[key] = md.get("title") or key
            doc_key = st.selectbox("Choose a document:", options=list(docs.keys()), format_func=lambda k: docs[k], key="sent_doc_key")
            doc_chunks = [c for c in st.session_state.es.chunks if (c.get("metadata",{}).get("path") or c.get("metadata",{}).get("title")) == doc_key]
            doc_chunks = sorted(doc_chunks, key=lambda c: (c.get("metadata",{}).get("page",0), c.get("chunk_id",0)))
            labels = [f"{i+1}. p{c['metadata'].get('page','?')} — {one_line_preview(c['text'], 80)}…" for i,c in enumerate(doc_chunks)]
            sel = st.multiselect("Select chunks to analyze:", options=list(range(len(doc_chunks))),
                                 format_func=lambda i: labels[i],
                                 default=list(range(len(doc_chunks))), key="sent_doc_sel")
            chosen = [doc_chunks[i] for i in sel]
            if chosen and st.button("Analyze selected", key="btn_sent_doc_sel"):
                results = [st.session_state.sent.analyze(c["text"]) for c in chosen]
                
                st.subheader("DistilBERT Analysis")
                distilbert_sentiments = [r["distilbert"]["sentiment"] for r in results]
                st.plotly_chart(sentiment_bar(distilbert_sentiments), use_container_width=True)
                a,b,c = st.columns(3)
                with a: st.metric("Positive", f"{distilbert_sentiments.count('positive')}/{len(chosen)}")
                with b: st.metric("Neutral", f"{distilbert_sentiments.count('neutral')}/{len(chosen)}")
                with c: st.metric("Negative", f"{distilbert_sentiments.count('negative')}/{len(chosen)}")
                
                st.subheader("RoBERTa Analysis")
                roberta_sentiments = [r["roberta"]["sentiment"] for r in results]
                st.plotly_chart(sentiment_bar(roberta_sentiments), use_container_width=True)
                d,e,f = st.columns(3)
                with d: st.metric("Positive", f"{roberta_sentiments.count('positive')}/{len(chosen)}")
                with e: st.metric("Neutral", f"{roberta_sentiments.count('neutral')}/{len(chosen)}")
                with f: st.metric("Negative", f"{roberta_sentiments.count('negative')}/{len(chosen)}")

    else:  # Custom Text
        # Allows sentiment analysis on free-form text input
        txt = st.text_area("Paste text to analyze sentiment", height=160, key="sent_custom_text")
        if st.button("Analyze custom text", key="btn_sent_custom"):
            if txt:
                res = st.session_state.sent.analyze(txt)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("DistilBERT")
                    st.metric("Sentiment", f"{res['distilbert']['sentiment']} ({res['distilbert']['stars']}★)")
                    st.caption(f"Confidence: {res['distilbert']['confidence']:.2f}")
                with col2:
                    st.subheader("RoBERTa")
                    st.metric("Sentiment", f"{res['roberta']['sentiment']} ({res['roberta']['stars']}★)")
                    st.caption(f"Confidence: {res['roberta']['confidence']:.2f}")
            else:
                st.warning("Please enter some text to analyze.")

# ---------------- Summarization Tab ----------------
with tabs[3]:
    st.header("📝 Summarization")

    model_label = st.selectbox("Summarization model", list(SUMM_MODELS.keys()),
                               index=0, key="summ_model_label")
    model_id = SUMM_MODELS[model_label]
    with st.spinner(f"Loading model: {model_label}"):
        sum_pipe = get_sum_pipe(model_id)

    smode = st.radio("Input",
                     ["Selected Chunks (Last Retrieval)", "Whole Document", "Selected Chunks (Pick Document)", "Custom Text"],
                     index=0, key="summ_mode")

    if smode == "Selected Chunks (Last Retrieval)":
        res = st.session_state.get("last_chunks", [])
        if not res:
            st.info("No recent retrieval. Ask a question in Q&A first.")
        else:
            labels = [f"{i+1}. {r['chunk']['metadata'].get('title','')} (p.{r['chunk']['metadata'].get('page','?')}) — {one_line_preview(r['chunk']['text'], 80)}…"
                      for i, r in enumerate(res)]
            sel = st.multiselect("Select chunks to summarize:", options=list(range(len(res))),
                                 format_func=lambda i: labels[i],
                                 default=list(range(len(res))), key="summ_last_sel")
            if sel and st.button("Summarize selected", key="btn_summ_sel"):
                # Concatenate selected chunks into a single context for summarization
                ctx = " ".join([res[i]["chunk"]["text"] for i in sel])
                summ = run_summarize_long(sum_pipe, ctx, model_label) if len(ctx) > 1200 else run_summarize(sum_pipe, ctx, model_label)
                scores = rouge_scores(ctx, summ)
                st.subheader("Summary"); st.success(summ)
                st.plotly_chart(rouge_bar(scores), use_container_width=True)
                # Log the metrics
                st.session_state.metrics_log["rouge_runs"].append(
                    {"timestamp": datetime.now().isoformat(), "model": model_label, "mode": "last_retrieved", **scores}
                )

    elif smode == "Whole Document":
        # Summarizes all chunks of a selected document
        if not st.session_state.ready or not st.session_state.es:
            st.info("Build the index first.")
        else:
            docs = {}
            for c in st.session_state.es.chunks:
                md = c.get("metadata", {})
                key = md.get("path") or md.get("title")
                docs[key] = md.get("title") or key
            doc_key = st.selectbox("Choose a document:", options=list(docs.keys()), format_func=lambda k: docs[k], key="summ_doc_key")
            if st.button("Summarize full document", key="btn_summ_doc"):
                doc_chunks = [c for c in st.session_state.es.chunks
                              if (c.get("metadata",{}).get("path") or c.get("metadata",{}).get("title")) == doc_key]
                doc_chunks = sorted(doc_chunks, key=lambda c: (c.get("metadata",{}).get("page",0), c.get("chunk_id",0)))
                ctx = " ".join([c["text"] for c in doc_chunks])
                summ = run_summarize_long(sum_pipe, ctx, model_label) if len(ctx) > 1200 else run_summarize(sum_pipe, ctx, model_label)
                scores = rouge_scores(ctx, summ)
                st.subheader("Document Summary"); st.success(summ)
                st.plotly_chart(rouge_bar(scores), use_container_width=True)
                with st.expander("Source chunks"):
                    for i, c in enumerate(doc_chunks, 1):
                        md = c["metadata"]
                        st.write(f"**Chunk {i}** — {md.get('title','')} (p.{md.get('page','?')})")
                        st.write(c["text"][:300] + "…")
                st.session_state.metrics_log["rouge_runs"].append(
                    {"timestamp": datetime.now().isoformat(), "model": model_label, "mode": "whole_document", **scores}
                )

    elif smode == "Selected Chunks (Pick Document)":
        # Summarizes specific chunks from a selected document
        if not st.session_state.ready or not st.session_state.es:
            st.info("Build the index first.")
        else:
            docs = {}
            for c in st.session_state.es.chunks:
                md = c.get("metadata", {})
                key = md.get("path") or md.get("title")
                docs[key] = md.get("title") or key
            doc_key = st.selectbox("Choose a document:", options=list(docs.keys()),
                                   format_func=lambda k: docs[k], key="summ_doc_pick_key")
            doc_chunks = [c for c in st.session_state.es.chunks
                          if (c.get("metadata",{}).get("path") or c.get("metadata",{}).get("title")) == doc_key]
            doc_chunks = sorted(doc_chunks, key=lambda c: (c.get("metadata",{}).get("page",0), c.get("chunk_id",0)))
            labels = [f"{i+1}. p{c['metadata'].get('page','?')} — {one_line_preview(c['text'], 80)}…" for i,c in enumerate(doc_chunks)]
            sel = st.multiselect("Select chunks:", options=list(range(len(doc_chunks))),
                                 format_func=lambda i: labels[i],
                                 default=list(range(len(doc_chunks))), key="summ_doc_sel")
            if sel and st.button("Summarize selected chunks", key="btn_summ_doc_sel"):
                ctx = " ".join([doc_chunks[i]["text"] for i in sel])
                summ = run_summarize_long(sum_pipe, ctx, model_label) if len(ctx) > 1200 else run_summarize(sum_pipe, ctx, model_label)
                scores = rouge_scores(ctx, summ)
                st.subheader("Summary"); st.success(summ)
                st.plotly_chart(rouge_bar(scores), use_container_width=True)
                st.session_state.metrics_log["rouge_runs"].append(
                    {"timestamp": datetime.now().isoformat(), "model": model_label, "mode": "picked_chunks", **scores}
                )

    else:  # Custom Text
        # Summarizes free-form text input
        txt = st.text_area("Paste text to summarize", height=220, key="custom_sum_text")
        if st.button("Summarize custom text", key="btn_summ_custom"):
            summ = run_summarize_long(sum_pipe, txt, model_label) if len(txt) > 1200 else run_summarize(sum_pipe, txt, model_label)
            scores = rouge_scores(txt, summ) if txt else {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            st.subheader("Summary"); st.success(summ)
            st.plotly_chart(rouge_bar(scores), use_container_width=True)
            st.session_state.metrics_log["rouge_runs"].append(
                    {"timestamp": datetime.now().isoformat(), "model": model_label, "mode": "custom_text", **scores}
                )

# ---------------- Metrics Tab ----------------
with tabs[4]:
    st.header("📈 Metrics")
    runs = st.session_state.metrics_log.get("rouge_runs", [])
    if not runs:
        st.info("Run a summarization to collect ROUGE per model/mode.")
    else:
        # Displays ROUGE metrics in a DataFrame and generates plots
        df = pd.DataFrame(runs)
        st.dataframe(df, use_container_width=True)
        for model_name, sub in df.groupby("model"):
            st.subheader(model_name)
            for metric in ["rouge1","rouge2","rougeL"]:
                fig = go.Figure()
                for mode, subm in sub.groupby("mode"):
                    fig.add_trace(go.Scatter(x=list(range(len(subm))), y=subm[metric], mode="lines+markers", name=mode))
                fig.update_layout(title=f"{metric} across runs ({model_name})", yaxis_range=[0,1],
                                  height=260, margin=dict(l=10,r=10,t=40,b=10), legend_title="Mode")
                st.plotly_chart(fig, use_container_width=True)