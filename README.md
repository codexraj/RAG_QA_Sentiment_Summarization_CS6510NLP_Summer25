# RAG_QA_Sentiment_Summarization_CS6510NLP_Summer25
Final NLP Project for Summer2025 Northeastern University
RAG + NLP PIPELINE (README.txt)
Authors: Rajorshi Sarkar, Priya Rupesh Mehta
Version: 1.0

───────────────────────────────────────────────────────────────────────────────
OVERVIEW
───────────────────────────────────────────────────────────────────────────────
This repository implements a fast, transparent Retrieval-Augmented Generation
(RAG) workflow with a Streamlit UI. It ingests PDFs, builds a FAISS index over
dense embeddings, retrieves top-k passages for a query, optionally diversifies
results (MMR / k-means) and/or re-ranks them using a cross-encoder, then runs:
  • Extractive Question Answering (BART SQuAD-style)
  • Summarization (single-pass for short text; map-reduce for long docs)
  • Lightweight Sentiment Analysis (positive/neutral/negative)

Key files:
  - rag_pipeline.py  → core pipeline: ingestion, embeddings, search, QA,
                       summarization, sentiment, reranking, diversification
  - frontend.py      → Streamlit dashboard for end-to-end control and preview

Primary goals:
  1) Accurate answers grounded in retrieved evidence (with citations)
  2) Robust, CPU-friendly defaults; GPU optional if available
  3) Simple UX: Preview → Ingest → Ask → Inspect chunks → Export

───────────────────────────────────────────────────────────────────────────────
FEATURES
───────────────────────────────────────────────────────────────────────────────
• PDF ingestion with text cleaning & overlap-aware chunking
• SentenceTransformer embeddings + FAISS index (cosine via L2-normalized vecs)
• Query-time retrieval with optional diversification:
    – Maximal Marginal Relevance (MMR)
    – k-means clustering (reduces redundancy)
• Optional cross-encoder re-ranker for semantic re-scoring
• Per-document context selection to avoid mixing multiple PDFs
• Extractive QA using a BART SQuAD-style reader
• Summarization:
    – Short text: single pass
    – Long text: map-reduce (chunk summaries → merged final)
• Sentiment (neg/neu/pos) for quick appraisal of passages or answers
• Disk cache + index manifest for reuse; one-click Clear Cache in UI
• Optional CORD-19 download (with Kaggle credentials) for experimentation

───────────────────────────────────────────────────────────────────────────────
ARCHITECTURE / DATA FLOW
───────────────────────────────────────────────────────────────────────────────
PDFs → Clean & Chunk → Embed ─┐
                              ├→ FAISS Index (persisted)
Query → Embed ─────────────────┘
         ↓
 Retrieve top-K → (MMR / k-means) → (Cross-encoder rerank?)
         ↓
  Assemble per-document context (bounded window)
         ↓
     ┌───────────────┬─────────────────────┐
     │ QA (BART)     │  Summarizer         │
     └───────────────┴─────────────────────┘
         ↓                     ↓
   Answer + citations     Short/Long summary

──────────────────────────────────────────────────────────────────────────────―
REQUIREMENTS
──────────────────────────────────────────────────────────────────────────────―
• Python 3.10+
• CPU: works out of the box (FAISS CPU). GPU optional (Torch CUDA/MPS).
• Recommended packages (see your environment/requirements):
    numpy, pandas, scikit-learn, faiss-cpu
    torch, transformers, sentence-transformers
    streamlit, plotly, tqdm, pyyaml, regex, python-dotenv (optional)
    (macOS) set PYTORCH_ENABLE_MPS_FALLBACK=1 if using MPS

───────────────────────────────────────────────────────────────────────────────
INSTALLATION
───────────────────────────────────────────────────────────────────────────────
# Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt  # or install the libs listed above

# (Optional) FAISS on GPU: use faiss-gpu instead of faiss-cpu if CUDA is ready

───────────────────────────────────────────────────────────────────────────────
RUNNING THE APP
───────────────────────────────────────────────────────────────────────────────
# Launch Streamlit UI
streamlit run frontend.py

Default URL: http://localhost:8501

Tips:
• If import fails, ensure rag_pipeline.py is in PYTHONPATH (frontend adds '.').
• For faster CPU inference, the code caps BLAS threads and disables CUDA by
  default (see environment settings at top of rag_pipeline.py).

───────────────────────────────────────────────────────────────────────────────
UI WORKFLOW
───────────────────────────────────────────────────────────────────────────────
1) PREVIEW
   - Upload PDFs (sidebar) and preview detected text/chunks.
   - (Optional) Enable CORD-19 download (Kaggle API key required).

2) INGEST
   - Click “Ingest” to compute embeddings and build/update the FAISS index.
   - Index reuse is tracked via a manifest hash to avoid re-embedding unchanged
     files.

3) ASK
   - Enter a query. Configure:
       • top-k passages
       • Diversification: MMR OR k-means
       • Cross-encoder reranker (on/off)
       • Rerank initial pool size (≥ k, rescored then truncated to k)
   - Inspect retrieved chunks + metadata/citations.

4) ANSWER/SUMMARIZE/SENTIMENT
   - QA: BART SQuAD-style extractive answers grounded in context.
   - Summarize: single-pass for short text; map-reduce for long documents.
   - Sentiment: quick neg/neu/pos scoring with stars (1/3/5).

5) CLEAR CACHE (optional)
   - Remove uploads & index artifacts via the sidebar Clear Cache.

───────────────────────────────────────────────────────────────────────────────
MODELS (DEFAULTS & NOTES)
───────────────────────────────────────────────────────────────────────────────
• Embeddings: SentenceTransformer model (configurable in code)
• Reader (QA): BART SQuAD-style model (extractive)
• Summarizer: BART-family summarization models; long-doc path uses map-reduce
• Sentiment: RoBERTa Twitter sentiment (LABEL_0/1/2 → neg/neu/pos mapping)

All model names are configurable; the code uses safe, public defaults.

───────────────────────────────────────────────────────────────────────────────
DIVERSIFICATION & RERANKING
───────────────────────────────────────────────────────────────────────────────
• MMR balances relevance vs diversity to reduce redundancy across hits.
• k-means clusters candidate passages; pick one representative per cluster.
• Cross-encoder rerank rescales a *larger* initial pool; after rescoring,
  results are truncated to top-k (post-rerank).

Best practice:
  - Use rerank_initial_k = max(25, 5*k)
  - Enable either MMR or k-means (not both) for clarity in evaluation.

───────────────────────────────────────────────────────────────────────────────
INDEX PERSISTENCE & MANIFEST
───────────────────────────────────────────────────────────────────────────────
• The pipeline persists FAISS index and metadata to disk.
• A manifest hash (of inputs + settings) prevents unnecessary recomputation.
• To hard-reset, use the Clear Cache button or delete the index artifacts. 

───────────────────────────────────────────────────────────────────────────────
OPTIONAL: CORD-19 DATA
───────────────────────────────────────────────────────────────────────────────
• Add Kaggle credentials in the sidebar (uploads kaggle.json securely).
• Choose a max-articles limit; ingestion uses the same pipeline.
• Respect dataset terms; only use for research/experimentation.

───────────────────────────────────────────────────────────────────────────────
ENVIRONMENT SETTINGS (DEFAULTS IN CODE)
───────────────────────────────────────────────────────────────────────────────
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
PYTORCH_ENABLE_MPS_FALLBACK=1
CUDA_VISIBLE_DEVICES=""   # forces CPU unless you override

These keep CPU inference stable and avoid oversubscription on laptops.

───────────────────────────────────────────────────────────────────────────────
TROUBLESHOOTING
───────────────────────────────────────────────────────────────────────────────
• ImportError: Ensure both files live together and Streamlit runs from that dir.
• Model load OSError: Check internet access or pre-download models to cache.
• Slow queries: reduce k, disable reranker, or use a smaller embedding model.
• Empty answers: verify chunks contain the needed span; increase top-k; try
  disabling diversification; check per-document context assembly.
• Shape errors in NumPy: make sure encode() uses convert_to_numpy=True.
• macOS Metal (MPS) quirks: keep PYTORCH_ENABLE_MPS_FALLBACK=1.

───────────────────────────────────────────────────────────────────────────────
SECURITY & PRIVACY
───────────────────────────────────────────────────────────────────────────────
• Uploaded PDFs are processed locally.
• If using third-party datasets/APIs, follow their licenses and terms.
• Consider redacting sensitive content before indexing.

───────────────────────────────────────────────────────────────────────────────
ROADMAP (SUGGESTED)
───────────────────────────────────────────────────────────────────────────────
• Export answers/summaries as Markdown/PDF
• Per-query model selection (fast vs accurate profiles)
• Doc-type adaptive chunking (tables, OCR layer, multilingual)
• Hybrid dense + keyword retrieval (BM25 + embeddings)

───────────────────────────────────────────────────────────────────────────────
LICENSE
───────────────────────────────────────────────────────────────────────────────
All rights reserved. © 2025 Rajorshi Sarkar & Priya Rupesh Mehta
