# -*- coding: utf-8 -*-
# rag_pipeline.py — Core logic for RAG, Sentiment, and Summarization
# CS6120 Project build
# 13 AUG 2025 - Rajorshi Sarkar / Priyan Rupesh Mehta
"""
RAG PIPELINE (RAG + Sentiment + Summarization)
- PDF ingestion with text cleaning
- SentenceTransformer embeddings + FAISS
- Optional cross-encoder re-ranker
- Per-document context selection to avoid multi-PDF confusion
- QA, Sentiment, Summarization utilities
"""

# ---- keep BLAS threads low & force CPU (avoid MPS OOM on macOS) ----
# Sets environment variables to limit thread usage and force CPU to prevent
# out-of-memory errors on certain systems.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure CPU-only

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import fitz  # PyMuPDF for PDF text extraction
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BartForConditionalGeneration, PegasusForConditionalGeneration, PegasusTokenizer
from rouge_score import rouge_scorer
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# ===================== Utilities =====================

def log(msg: str):
    """Simple logging function to print messages to the console."""
    print(msg, flush=True)

def setup_kaggle_credentials(username: str, key: str):
    """Sets up Kaggle API credentials for downloading datasets."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    cred_path = kaggle_dir / "kaggle.json"
    with open(cred_path, "w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f)
    try:
        cred_path.chmod(0o600)
    except Exception:
        pass
    log(f"✅ Kaggle credentials set at {cred_path}")

# ---------------------- text cleaning ----------------------
def clean_text(txt: str) -> str:
    """Fix common PDF artifacts such as hyphens at line breaks and multiple spaces."""
    if not txt:
        return ""
    txt = txt.replace("\r", "\n")
    txt = txt.replace("\xad", "")          # soft hyphen
    txt = txt.replace("\u00a0", " ")       # NBSP (non-breaking space)
    txt = re.sub(r"-\s*\n\s*", "", txt)    # Remove hyphenation at line breaks
    txt = re.sub(r"(?<=\w)\n(?=\w)", " ", txt)  # Replace newlines between words with a space
    txt = re.sub(r"[ \t]{2,}", " ", txt)   # Consolidate multiple spaces/tabs
    txt = re.sub(r"\n{2,}", "\n\n", txt)   # Consolidate multiple newlines
    return txt.strip()

# New text cleaning function for model output
def clean_model_output(txt: str) -> str:
    """Robust text cleaning for generated model output to fix common formatting issues."""
    if not txt:
        return ""
    txt = txt.strip()
    
    # Fix incomplete hyphenated words from truncation
    txt = re.sub(r"(\w+)-(\s*\w+)", r"\1\2", txt)
    
    # Replace common tokenization artifacts
    txt = txt.replace("multip le", "multiple")
    
    # Fix spacing issues with common acronyms and technical terms
    txt = re.sub(r"(?i)systemonachip", "System-on-a-chip", txt)
    txt = re.sub(r"So C", "SoC", txt)
    txt = re.sub(r"onchip", "on-chip", txt)
    
    # Correct the specific "highfrequency" issue
    txt = txt.replace("highfrequency", "high frequency")
    
    # Fix spaces around punctuation
    txt = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", txt)
    
    # Ensure only a single space between words
    txt = re.sub(r"\s+", " ", txt)
    
    # Capitalize the first letter of the sentence
    if len(txt) > 0:
        txt = txt[0].upper() + txt[1:]
    
    # Fix misplaced parenthesis for common terms like IP
    txt = re.sub(r"IP\)cores\)", "IP cores)", txt)
    
    return txt.strip()

# ===================== PDF ingestion =====================

def ingest_pdfs(paths_or_dir, max_pages: Optional[int] = None) -> List[Dict]:
    """
    Extracts text from PDFs using PyMuPDF and returns a list of dictionaries,
    with each dictionary representing a page.
    """
    records: List[Dict] = []
    if paths_or_dir is None:
        return records

    # Normalize to a list of Paths
    if isinstance(paths_or_dir, (str, Path)):
        p = Path(paths_or_dir)
        if p.is_dir():
            pdf_paths = sorted(p.glob("**/*.pdf"))
        elif p.suffix.lower() == ".pdf":
            pdf_paths = [p]
        else:
            raise ValueError(f"Unsupported path for PDFs: {paths_or_dir}")
    else:
        pdf_paths = [Path(x) for x in paths_or_dir]

    for pdf in pdf_paths:
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            log(f"[WARN] Skipping {pdf}: {e}")
            continue

        title = (doc.metadata or {}).get("title") or pdf.stem
        doc_id = f"pdf::{pdf.name}"

        for page_idx in range(len(doc)):
            if max_pages is not None and page_idx >= max_pages:
                break
            try:
                page = doc.load_page(page_idx)
                text = clean_text(page.get_text("text"))
            except Exception as e:
                log(f"[WARN] Failed reading page {page_idx+1} of {pdf}: {e}")
                text = ""
            if not text or text.isspace():
                continue
            # Optional light header/footer trim to remove common page numbers/footers
            lines = [ln.strip() for ln in text.splitlines()]
            if len(lines) > 3 and len(lines[0]) <= 40:
                lines = lines[1:]
            if len(lines) > 3 and len(lines[-1]) <= 40:
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if not text:
                continue
            records.append({
                "text": text,
                "metadata": {
                    "source": "pdf",
                    "doc_id": doc_id,
                    "title": title,
                    "page": page_idx + 1,
                    "path": str(pdf.resolve()),
                }
            })
        doc.close()
    log(f"✅ Ingested {len(records)} PDF page-records from {len(pdf_paths)} file(s)")
    return records

# ===================== Datasets =====================

class DatasetManager:
    """Handles downloading and processing external datasets like arXiv, GovReport, and CORD-19."""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def _have_kaggle(self) -> bool:
        """Checks if Kaggle credentials are set up."""
        try:
            import kaggle  # noqa: F401
            kg_home = Path.home() / ".kaggle" / "kaggle.json"
            return kg_home.exists() or all(k in os.environ for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"])
        except Exception:
            return False

    def download_arxiv(self, limit=500, keywords="", categories="cs.CL,cs.AI,cs.LG",
                       date_from: Optional[str]=None, date_to: Optional[str]=None) -> List[Dict]:
        """Downloads and processes arXiv metadata from Kaggle."""
        log("\n📥 [arXiv] Downloading/processing via Kaggle...")
        if not self._have_kaggle():
            log("   ⚠️ Kaggle credentials not found. Skipping arXiv.")
            return []
        try:
            import kaggle
            out = Path(self.data_dir) / "arxiv"
            out.mkdir(parents=True, exist_ok=True)
            kaggle.api.dataset_download_files('Cornell-University/arxiv', path=str(out), unzip=True)
            json_path = out / "arxiv-metadata-oai-snapshot.json"
            if not json_path.exists():
                log("   ⚠️ arxiv-metadata file not found; skipping.")
                return []

            kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
            cat_set = set([c.strip() for c in categories.split(",") if c.strip()])
            from_dt = pd.to_datetime(date_from) if date_from else None
            to_dt = pd.to_datetime(date_to) if date_to else None

            docs: List[Dict] = []
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(docs) >= limit:
                        break
                    try:
                        paper = json.loads(line)
                        title = str(paper.get('title', ''))
                        abstract = str(paper.get('abstract', ''))
                        cats = set(str(paper.get('categories', '')).split())

                        if kw_list and not any(k in title.lower() or k in abstract.lower() for k in kw_list):
                            continue
                        if cat_set and not (cats & cat_set):
                            continue
                        upd = paper.get('update_date') or (paper.get('versions') or [{}])[0].get('created')
                        if (from_dt or to_dt) and upd:
                            try:
                                dt = pd.to_datetime(upd)
                                if from_dt and dt < from_dt: continue
                                if to_dt and dt > to_dt: continue
                            except Exception:
                                pass

                        if abstract:
                            docs.append({
                                "text": f"Title: {title}\n\nAbstract: {abstract}",
                                "metadata": {"title": title, "source": "arXiv",
                                             "categories": paper.get('categories', '')}
                            })
                    except Exception:
                        continue
            log(f"   ✅ Processed {len(docs)} arXiv entries")
            return docs
        except Exception as e:
            log(f"   ⚠️ Error arXiv: {e}")
            return []

    def download_govreport(self, limit: int = 100, keywords: str = "") -> List[Dict]:
        """Loads the GovReport dataset from HuggingFace Hub."""
        log("\n📥 [GovReport] Loading from HuggingFace...")
        try:
            from datasets import load_dataset
            ds = load_dataset("ccdv/govreport-summarization", split=f"train[:{limit}]")
            kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
            docs = []
            for item in ds:
                title = str(item.get("title", "GovReport Document"))
                report = str(item.get("report") or "")
                if not report:
                    continue
                if kw_list:
                    text_for_filter = (title + " " + report).lower()
                    if not any(k in text_for_filter for k in kw_list):
                        continue
                docs.append({"text": report[:5000],
                             "metadata": {"title": title, "source": "GovReport"}})
            log(f"   ✅ Loaded {len(docs)} GovReport docs")
            return docs
        except Exception as e:
            log(f"   ⚠️ Error GovReport: {e}")
            return []

    def download_cord19(self, limit: int = 1000) -> List[Dict]:
        """Downloads and processes CORD-19 metadata from Kaggle."""
        log("\n📥 [CORD-19] Downloading/processing...")
        if not self._have_kaggle():
            log("   ⚠️ Kaggle credentials not found. Skipping CORD-19.")
            return []
        try:
            import kaggle, zipfile
            out = Path(self.data_dir) / "cord19"
            out.mkdir(parents=True, exist_ok=True)
            kaggle.api.dataset_download_file('allen-institute-for-ai/CORD-19-research-challenge', 'metadata.csv', path=str(out), force=True)
            zip_path = out / 'metadata.csv.zip'
            csv_path = out / 'metadata.csv'
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(out)
                zip_path.unlink()
            if not csv_path.exists():
                log("   ⚠️ metadata.csv not found; skipping.")
                return []
            docs: List[Dict] = []
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=enc, on_bad_lines='skip', low_memory=False, nrows=5000)
                    log(f"   ✓ Read CSV with {enc}")
                    break
                except Exception as e:
                    log(f"   Failed with {enc}: {str(e)[:60]}")
            if df is None:
                log("   ⚠️ Could not read metadata.csv with common encodings.")
                return []
            for _, row in df.iterrows():
                abstract = str(row.get('abstract', '') or '')
                title = str(row.get('title', '') or '')
                abstract = ''.join(ch for ch in abstract if ord(ch) < 128 or ch.isspace())
                title = ''.join(ch for ch in title if ord(ch) < 128 or ch.isspace())
                if abstract and len(abstract) > 100:
                    docs.append({
                        "text": f"Title: {title}\n\nAbstract: {abstract}",
                        "metadata": {"title": title, "source": "CORD-19",
                                     "authors": str(row.get('authors', '') or '')[:200]}
                    })
                if len(docs) >= limit:
                    break
            log(f"   ✅ Processed {len(docs)} CORD-19 entries")
            return docs
        except Exception as e:
            log(f"   ⚠️ Error CORD-19: {e}")
            return []

    def build_corpus(self, include_datasets=True, include_cord19=False, pdf_dir: Optional[str]=None,
                     save_json: str="data/all_documents.json",
                     arxiv_max: int=500, gov_max: int=100,
                     arxiv_keywords: str="", arxiv_categories: str="cs.CL,cs.AI,cs.LG",
                     arxiv_date_from: Optional[str]=None, arxiv_date_to: Optional[str]=None,
                     gov_keywords: str="", interactive_select: bool=False) -> List[Dict]:
        """
        Builds the complete corpus from specified sources (datasets and local PDFs).
        Saves the manifest to a JSON file.
        """
        docs: List[Dict] = []

        if include_datasets:
            docs += self.download_arxiv(limit=arxiv_max, keywords=arxiv_keywords,
                                        categories=arxiv_categories,
                                        date_from=arxiv_date_from, date_to=arxiv_date_to)
            docs += self.download_govreport(limit=gov_max, keywords=gov_keywords)
            if include_cord19:
                docs += self.download_cord19(limit=1000)

        if pdf_dir:
            docs += ingest_pdfs(pdf_dir)

        if not docs:
            log("\n⚠️ No external data available. Using sample documents...")
            docs = [
                {"text": "Intro to Machine Learning ...", "metadata": {"title": "ML Intro", "source": "Sample"}},
                {"text": "NLP fundamentals ...", "metadata": {"title": "NLP Fundamentals", "source": "Sample"}},
                {"text": "Retrieval-Augmented Generation (RAG) ...", "metadata": {"title": "RAG Basics", "source": "Sample"}},
            ]

        # Persist manifest
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        log(f"\n💾 Saved manifest to: {save_json}")
        return docs

# ===================== Embeddings + FAISS (with re-ranker) =====================

class EmbeddingSystem:
    """
    Manages document chunking, embedding generation with SentenceTransformers,
    and vector indexing with FAISS. Also includes a cross-encoder re-ranker.
    """
    # default retriever (CPU friendly, good quality)
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        log(f"\n🔧 Initializing Embedding System: {model_name}")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings = None
        self.index = None
        self._reranker_loaded = False
        self._rerank_pair_len = 384  # default truncation

    # ---------- chunking ----------
    def create_chunks(self, documents, *, max_chars_per_page: int = 2000, chunks_per_page: int = 1):
        """Uniformly splits each document page into `chunks_per_page` segments."""
        all_chunks = []
        total = len(documents)
        for i, doc in enumerate(documents):
            print(f"[chunking] {i+1}/{total}", flush=True)
            text = (doc.get("text") or "").strip()
            if not text:
                continue
            text = text[:max_chars_per_page]
            md = doc.get("metadata", {}) or {}
            n = max(1, int(chunks_per_page))
            step = len(text) // n
            for cidx in range(n):
                start = cidx * step
                end = start + step if cidx < n - 1 else len(text)
                seg = text[start:end]
                if not seg:
                    continue
                all_chunks.append({
                    "id": f"{md.get('doc_id', f'doc_{i}')}_p{md.get('page',0)}_c{cidx}",
                    "text": seg,
                    "doc_id": md.get("doc_id", i),
                    "chunk_id": cidx,
                    "metadata": md
                })
        self.chunks = all_chunks
        log(f"✅ Created {len(all_chunks)} chunks (uniform split per page, N={chunks_per_page})")
        return all_chunks

    # ---------- embeddings / faiss ----------
    def generate_embeddings(self, batch_size: int = 2):
        """Generates embeddings for all chunks using the SentenceTransformer model."""
        if self.chunks is None or not self.chunks:
            raise ValueError("No chunks available. Call create_chunks first.")
        texts = [c["text"] for c in self.chunks]
        log(f"\n🧮 Embedding {len(texts)} chunks ...")
        self.embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        log(f"✅ Embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def create_faiss_index(self):
        """Creates a FAISS index from the generated embeddings for efficient search."""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Call generate_embeddings first.")
        dim = self.embeddings.shape[1]
        faiss.normalize_L2(self.embeddings)  # Use dot product for cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype("float32"))
        log(f"✅ FAISS index size: {self.index.ntotal}")
        return self.index

    def search(self, query: str, k: int = 5):
        """Performs a semantic search against the FAISS index."""
        qe = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(qe)
        scores, idxs = self.index.search(qe.astype("float32"), k)
        results = []
        for idx, sc in zip(idxs[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                results.append({"chunk": self.chunks[idx], "score": float(sc)})
        return results

    def save(self, index_path="faiss_index.bin", chunks_path="chunks.pkl"):
        """Saves the FAISS index and chunk metadata to disk."""
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        log("💾 Saved FAISS index and chunks.")

    def load(self, index_path="faiss_index.bin", chunks_path="chunks.pkl"):
        """Loads the FAISS index and chunk metadata from disk."""
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        log(f"📂 Loaded index, vectors: {self.index.ntotal}")

    # ---------- re-ranker (cross-encoder) ----------
    def _ensure_reranker(self, max_pair_len: int = 384):
        """
        Lazily loads the cross-encoder re-ranker model and tokenizer.
        """
        if self._reranker_loaded:
            self._rerank_pair_len = max_pair_len
            return
        log("\n⚖️  Loading cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2)…")
        self._tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._rerank_model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._reranker_loaded = True
        self._rerank_pair_len = max_pair_len
        log("✅ Re-ranker ready.")

    def rerank(self, query: str, results: List[Dict], *, top_m: int = 25, max_pair_len: int = 384) -> List[Dict]:
        """
        Re-scores a list of retrieved chunks using a cross-encoder model
        and returns the results sorted by the new score.
        """
        if not results:
            return []
        self._ensure_reranker(max_pair_len=max_pair_len)
        import torch
        keep = results[:top_m]
        pairs = [(query, r["chunk"]["text"]) for r in keep]
        with torch.no_grad():
            enc = self._tok([q for q,_ in pairs], [c for _,c in pairs],
                            padding=True, truncation=True, max_length=self._rerank_pair_len,
                            return_tensors="pt")
            logits = self._rerank_model(**enc).logits.squeeze(-1)
            scores = logits.tolist() if hasattr(logits, "tolist") else list(logits)
        ranked = sorted(zip(keep, scores), key=lambda x: x[1], reverse=True)
        return [r for r,_ in ranked]

    # ---------- doc grouping / context building ----------
    @staticmethod
    def doc_key(md: Dict[str, Any]) -> str:
        """Generates a unique key for a document from its metadata."""
        return md.get("path") or md.get("title") or md.get("doc_id") or "unknown"

    def group_by_doc(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Groups search results by their source document."""
        by: Dict[str, List[Dict]] = {}
        for r in results:
            key = self.doc_key(r["chunk"]["metadata"])
            by.setdefault(key, []).append(r)
        return by

    def top_doc_context(self, results: List[Dict], *, per_doc_top: int = 3) -> Tuple[str, List[Dict], str]:
        """
        Selects the best document based on aggregate scores and returns its top chunks
        as a concatenated string for context.
        """
        by = self.group_by_doc(results)
        if not by:
            return "", [], ""
        # score each doc by sum of candidate scores
        ranked_docs = sorted(by.items(), key=lambda kv: sum(x.get("score", 0.0) for x in kv[1]), reverse=True)
        best_key, items = ranked_docs[0]
        items = sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)[:max(1, per_doc_top)]
        ctx = " ".join([it["chunk"]["text"] for it in items])
        return best_key, items, ctx

# ===================== QA / Sentiment / Summarizer =====================

class QASystem:
    """Handles question-answering using multiple models and a retrieval system."""
    def __init__(self, embedding_system: EmbeddingSystem):
        self.embedding_system = embedding_system
        self.models = {}
        self._bart_tokenizer = None

    def load_models(self):
        """Loads the question-answering models from HuggingFace Hub."""
        log("\n🚀 Loading QA models ...")
        self.models["distilbert"] = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        self.models["roberta"] = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.models["bart_qa"] = pipeline("question-answering", model="phiyodr/bart-large-finetuned-squad2")
        self.models["t5"] = pipeline("text2text-generation", model="google/flan-t5-base")
        self._bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        log("✅ QA models loaded.")

    def search_diversified(self, query: str, k: int, rerank_enabled: bool, rerank_initial_k: int,
                           mmr_enabled: bool, mmr_lambda: float, kmeans_enabled: bool, kmeans_k: int):
        """
        Performs retrieval with optional reranking and diversity-focused
        strategies like MMR or K-means clustering.
        """
        
        # 1. Initial retrieval
        # Passes rerank_initial_k for the initial search when reranker is enabled.
        initial_results = self.embedding_system.search(query, k=rerank_initial_k if rerank_enabled else k)

        # 2. Reranking (if enabled)
        if rerank_enabled:
            # Passes the full initial pool to the reranker, then truncates to the final 'k'.
            ranked_results = self.embedding_system.rerank(
                query, initial_results, top_m=len(initial_results)
            )[:k]
        else:
            ranked_results = initial_results

        # Computes embeddings and query similarity here for use in diversity methods.
        if mmr_enabled or kmeans_enabled:
            embeddings = np.stack([
                self.embedding_system.model.encode(r['chunk']['text'], convert_to_numpy=True)
                for r in ranked_results
            ], axis=0)
            query_embedding = self.embedding_system.model.encode([query], convert_to_numpy=True)[0]
            sim_query = np.dot(embeddings, query_embedding)
            
        # 3. Diversity (if enabled)
        if mmr_enabled:
            # MMR implementation based on retrieved embeddings
            selected_indices = []
            while len(selected_indices) < k and len(selected_indices) < len(ranked_results):
                rem_indices = list(set(range(len(ranked_results))) - set(selected_indices))
                if not rem_indices: break
                
                max_score = -1
                best_idx = -1
                for i in rem_indices:
                    if selected_indices:
                        sim_rem = np.dot(embeddings[i], embeddings[selected_indices].T).max()
                    else:
                        sim_rem = 0
                    
                    mmr_score = mmr_lambda * sim_query[i] - (1-mmr_lambda) * sim_rem
                    if mmr_score > max_score:
                        max_score = mmr_score
                        best_idx = i
                
                if best_idx != -1:
                    selected_indices.append(best_idx)
            return [ranked_results[i] for i in selected_indices]
        
        if kmeans_enabled:
            num_clusters = min(kmeans_k, len(ranked_results))
            if num_clusters < 2: return ranked_results[:k]
            
            kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, n_init=10)
            kmeans.fit(embeddings)
            
            cluster_centers_embeddings = kmeans.cluster_centers_
            distances = np.dot(cluster_centers_embeddings, query_embedding)
            top_cluster_indices = np.argsort(distances)[-k:]
            
            diversified_results = []
            for cluster_idx in top_cluster_indices:
                cluster_chunks_indices = np.where(kmeans.labels_ == cluster_idx)[0]
                if len(cluster_chunks_indices) > 0:
                    top_chunk_in_cluster_idx = cluster_chunks_indices[np.argmax(sim_query[cluster_chunks_indices])]
                    diversified_results.append(ranked_results[top_chunk_in_cluster_idx])
            
            return diversified_results

        return ranked_results[:k]

    def answer(self, question: str, model_name: str = "all", doc_filter: Optional[str] = None,
               k: int = 5, rerank_enabled: bool = False, rerank_initial_k: int = 25,
               mmr_enabled: bool = False, mmr_lambda: float = 0.5,
               kmeans_enabled: bool = False, kmeans_k: int = 3) -> Dict:
        """
        Retrieves context, then generates answers using multiple QA models.
        """
        
        # 1. Search for chunks using selected strategy
        retrieved_chunks = self.search_diversified(
            query=question, k=k, rerank_enabled=rerank_enabled, rerank_initial_k=rerank_initial_k,
            mmr_enabled=mmr_enabled, mmr_lambda=mmr_lambda,
            kmeans_enabled=kmeans_enabled, kmeans_k=kmeans_k
        )

        # 2. Get context and top document
        chosen_doc, chosen, context = self.embedding_system.top_doc_context(retrieved_chunks, per_doc_top=k)
        
        out = {
            "question": question,
            "doc": chosen_doc,
            "answers": {},
            "retrieved": chosen,
            "context_preview": context[:400]
        }
        
        # Define a default message for when no answer is found
        default_no_answer_msg = "Sorry, I cannot find an answer to your question in the provided documents."

        def _get_sentence_from_answer(full_text, answer_span):
            """Finds and returns the full sentence containing the answer span."""
            if not answer_span:
                return ""
            # Find the start and end of the answer span within the full text
            start_idx = full_text.find(answer_span)
            if start_idx == -1:
                return answer_span
            
            end_idx = start_idx + len(answer_span)
            
            # Find the start of the sentence (look for previous period/question mark)
            sentence_start = full_text.rfind('.', 0, start_idx)
            if sentence_start == -1:
                sentence_start = 0
            else:
                sentence_start += 1 # Move past the period
            
            # Find the end of the sentence (look for next period/question mark)
            sentence_end = full_text.find('.', end_idx)
            if sentence_end == -1:
                sentence_end = len(full_text)
            else:
                sentence_end += 1 # Include the period
            
            return full_text[sentence_start:sentence_end].strip()

        # 3. Generate answers from models
        if model_name in ("all", "distilbert"):
            ans = self.models["distilbert"](question=question, context=context[:2000])
            if ans.get("score", 0.0) < 0.05 or not ans.get("answer", "").strip():
                out["answers"]["distilbert"] = default_no_answer_msg
            else:
                full_ans = _get_sentence_from_answer(context[:2000], ans.get("answer", ""))
                out["answers"]["distilbert"] = clean_model_output(full_ans if full_ans else ans.get("answer", ""))

        if model_name in ("all", "roberta"):
            ans = self.models["roberta"](question=question, context=context[:2000])
            if ans.get("score", 0.0) < 0.01 or not ans.get("answer", "").strip():
                out["answers"]["roberta"] = default_no_answer_msg
            else:
                full_ans = _get_sentence_from_answer(context[:2000], ans.get("answer", ""))
                out["answers"]["roberta"] = clean_model_output(full_ans if full_ans else ans.get("answer", ""))
        
        if model_name in ("all", "bart_qa"):
            ans = self.models["bart_qa"](question=question, context=context[:4000])
            full_ans = _get_sentence_from_answer(context[:2000], ans.get("answer", ""))
            out["answers"]["bart_qa"] = clean_model_output(full_ans if full_ans else ans.get("answer", ""))

        if model_name in ("all", "t5"):
            inp = f"question: {question} context: {context[:2000]}"
            t = self.models["t5"](inp, max_length=150)
            generated_text = t[0].get("generated_text")
            if len(generated_text.strip()) < 50:
                out["answers"]["t5"] = default_no_answer_msg
            else:
                generated_text = clean_model_output(generated_text)
                out["answers"]["t5"] = generated_text

        return out

class SentimentAnalyzer:
    """Analyzes the sentiment of a given text using multiple models."""
    def __init__(self):
        log("\n🎭 Loading sentiment models ...")
        self.pipes = {
            "distilbert": pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"),
            "roberta": pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        }
        log("✅ Sentiment analyzers ready.")
    def analyze(self, text: str) -> Dict:
        """Returns sentiment analysis results from each configured model."""
        results = {}
        for name, pipe in self.pipes.items():
            res = pipe(text[:512])[0]
            label = res["label"].lower()
            if name == "roberta":
                map_ = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}
                sentiment = map_.get(label, "neutral")
            else: # distilbert
                sentiment = "positive" if "pos" in label else ("negative" if "neg" in label else "neutral")
            stars = 5 if sentiment == "positive" else (1 if sentiment == "negative" else 3)
            results[name] = {"sentiment": sentiment, "stars": stars, "confidence": float(res["score"])}
        return results

class Summarizer:
    """Provides text summarization using multiple models, including a multi-pass approach for long texts."""
    def __init__(self):
        log("\n📝 Loading summarizer (DistilBART) ...")
        self.pipe_distilbart = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        # Add Pegasus model initialization
        log("📝 Loading summarizer (Pegasus) ...")
        self.pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        self.pipe_pegasus = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
        
        self._rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        log("✅ Summarizer ready.")

    def summarize(self, text: str, model_name: str = "distilbart", **kwargs) -> str:
        """Generates a summary for a single text input (up to 1024 tokens)."""
        if not text:
            return ""
        
        if model_name == "pegasus":
            default_params = {
                "max_length": 150,
                "min_length": 30,
                "length_penalty": 2.0,
                "num_beams": 4,
                "no_repeat_ngram_size": 3,
                "do_sample": False
            }
            params = {**default_params, **kwargs}
            
            inputs = self.pegasus_tokenizer(text[:1024], return_tensors="pt", truncation=True)
            summary_ids = self.pipe_pegasus.generate(inputs["input_ids"], **params)
            summary_text = self.pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary_text
        
        else: # "distilbart"
            out = self.pipe_distilbart(text[:1024], max_length=150, min_length=30)
            return out[0].get("summary_text", out[0].get("generated_text", ""))

    def summarize_long(self, text: str, model_name: str = "distilbart", *, window_chars: int = 1000) -> str:
        """
        Summarizes long text by breaking it into chunks, summarizing each,
        and then summarizing the resulting summaries.
        """
        if not text:
            return ""
        chunks = [text[i:i+window_chars] for i in range(0, len(text), window_chars)]
        partials = [self.summarize(c, model_name=model_name) for c in chunks]
        return self.summarize(" ".join(partials), model_name=model_name)
    
    def rouge(self, original: str, summary: str) -> Dict[str, float]:
        """Calculates ROUGE scores for a summary."""
        scores = self._rouge.score(original[:500], summary)
        return {k: v.fmeasure for k, v in scores.items()}

# ===================== Orchestrator =====================

class RAGPipeline:
    """Main orchestrator class to set up and run the entire RAG pipeline."""
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.embedding_system: Optional[EmbeddingSystem] = None
        self.qa_system: Optional[QASystem] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.summarizer: Optional[Summarizer] = None

    def setup(self,
              include_datasets: bool = True,
              include_cord19: bool = False,
              pdf_dir: Optional[str] = None,
              rebuild_index: bool = True,
              chunks_per_page: int = 1,
              arxiv_max: int = 500,
              gov_max: int = 100,
              arxiv_keywords: str = "",
              arxiv_categories: str = "cs.CL,cs.AI,cs.LG",
              arxiv_date_from: Optional[str] = None,
              arxiv_date_to: Optional[str] = None,
              gov_keywords: str = "",
              interactive_select: bool = False,
              use_reranker: bool = True,
              initial_k: int = 25,
              per_doc_top: int = 3,
              rerank_maxlen: int = 384):
        """Sets up the entire RAG pipeline from scratch or by loading a saved index."""
        log("\n" + "=" * 80)
        log("🚀 SETTING UP RAG PIPELINE")
        log("=" * 80)

        documents = self.dataset_manager.build_corpus(
            include_datasets=include_datasets,
            include_cord19=include_cord19,
            pdf_dir=pdf_dir,
            save_json="data/all_documents.json",
            arxiv_max=arxiv_max,
            gov_max=gov_max,
            arxiv_keywords=arxiv_keywords,
            arxiv_categories=arxiv_categories,
            arxiv_date_from=arxiv_date_from,
            arxiv_date_to=arxiv_date_to,
            gov_keywords=gov_keywords,
            interactive_select=interactive_select
        )

        if not documents:
            log("❎ No documents to index. Exiting setup.")
            return

        self.embedding_system = EmbeddingSystem()
        if rebuild_index or not (Path("faiss_index.bin").exists() and Path("chunks.pkl").exists()):
            self.embedding_system.create_chunks(documents, max_chars_per_page=2000, chunks_per_page=chunks_per_page)
            self.embedding_system.generate_embeddings(batch_size=2)
            self.embedding_system.create_faiss_index()
            self.embedding_system.save()
        else:
            self.embedding_system.load()

        self.qa_system = QASystem(self.embedding_system)
        self.qa_system.load_models()

        self.sentiment_analyzer = SentimentAnalyzer()
        self.summarizer = Summarizer()

        log("\n✅ Pipeline setup complete.")

    def run_interactive(self):
        """A simple interactive command-line interface for the RAG pipeline."""
        log("\n" + "=" * 80)
        log("🎮 INTERACTIVE RAG SYSTEM")
        log("=" * 80)
        while True:
            print("\n📋 Menu:\n1. Search (Embeddings)\n2. Q&A\n3. Sentiment on Top Chunks\n4. Summarize Top Chunks\n0. Exit")
            choice = input("\n> Choose option: ").strip()
            if choice == '0':
                log("👋 Goodbye!")
                break
            elif choice == '1':
                q = input("Query: ").strip()
                res = self.embedding_system.search(q, k=8)
                for i, r in enumerate(res, 1):
                    md = r['chunk']['metadata']
                    print(f"\n{i}. Score {r['score']:.4f} | src={md.get('source','?')} | page={md.get('page','-')} | title={md.get('title','-')}")
                    print(r['chunk']['text'][:300], "...")
            elif choice == '2':
                q = input("Question: ").strip()
                out = self.qa_system.answer(q, model_name="all")
                print(f"\n[DOC] {out.get('doc','')}")
                print("\nAnswers:")
                for m, a in out["answers"].items():
                    print(f"\n[{m.upper()}]\n{a}")
            elif choice == '3':
                q = input("Query (to fetch chunks): ").strip()
                res = self.embedding_system.search(q, k=5)
                print("\nSentiments:")
                for r in res:
                    s = self.sentiment_analyzer.analyze(r['chunk']['text'])
                    md = r['chunk']['metadata']
                    print(f"- DistilBERT: {s['distilbert']['sentiment']} ({s['distilbert']['stars']}★, conf={s['distilbert']['confidence']:.2f})")
                    print(f"- RoBERTa: {s['roberta']['sentiment']} ({s['roberta']['stars']}★, conf={s['roberta']['confidence']:.2f}) | src={md.get('source')} p={md.get('page','-')}")
            elif choice == '4':
                q = input("Query (to fetch chunks): ").strip()
                res = self.embedding_system.search(q, k=8)
                _, chosen, ctx = self.embedding_system.top_doc_context(res, per_doc_top=3)
                summ = self.summarizer.summarize_long(ctx) if len(ctx) > 1200 else self.summarizer.summarize(ctx)
                print("\nSummary:\n", summ)

# ===================== Main Execution Block =====================

def main():
    """Main function to parse arguments and set up the RAG pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="RAG Pipeline (+ re-ranker, per-doc context)")
    parser.add_argument("--pdf_dir", type=str, default=None, help="Directory of PDFs to ingest")
    parser.add_argument("--include-datasets", action="store_true", help="Include arXiv + GovReport datasets")
    parser.add_argument("--include-cord19", action="store_true", help="Include CORD-19 dataset")
    parser.add_argument("--interactive-select", action="store_true", help="Confirm before embedding/indexing")
    parser.add_argument("--set-kaggle-creds", nargs=2, metavar=("USERNAME","API_KEY"), help="Write Kaggle credentials")
    parser.add_argument("--arxiv-keywords", type=str, default="")
    parser.add_argument("--arxiv-categories", type=str, default="cs.CL,cs.AI,cs.LG")
    parser.add_argument("--arxiv-date-from", type=str, default=None)
    parser.add_argument("--arxiv-date-to", type=str, default=None)
    parser.add_argument("--arxiv-max", type=int, default=500)
    parser.add_argument("--gov-keywords", type=str, default="")
    parser.add_argument("--gov-max", type=int, default=100)
    parser.add_argument("--chunks-per-page", type=int, default=1, help="Uniform chunks per PDF page")

    # New QA/rerank knobs
    parser.add_argument("--use-reranker", action="store_true", help="Enable cross-encoder re-ranking")
    parser.add_argument("--initial-k", type=int, default=25, help="Pool size retrieved from FAISS before re-ranking")
    parser.add_argument("--per-doc-top", type=int, default=3, help="Top chunks from the selected document")
    parser.add_argument("--rerank-maxlen", type=int, default=384, help="Token max length for cross-encoder pairs")

    args = parser.parse_args()

    if args.set_kaggle_creds:
        setup_kaggle_credentials(args.set_kaggle_creds[0], args.set_kaggle_creds[1])

    dm = DatasetManager()
    docs = dm.build_corpus(
        include_datasets=args.include_datasets,
        include_cord19=args.include_cord19,
        pdf_dir=args.pdf_dir,
        arxiv_max=args.arxiv_max,
        gov_max=args.gov_max,
        arxiv_keywords=args.arxiv_keywords,
        arxiv_categories=args.arxiv_categories,
        arxiv_date_from=args.arxiv_date_from,
        arxiv_date_to=args.arxiv_date_to,
        gov_keywords=args.gov_keywords,
        interactive_select=args.interactive_select
    )

    if not docs:
        log("❎ No documents to index. Exiting setup.")
        return

    es = EmbeddingSystem()
    es.create_chunks(docs, max_chars_per_page=2000, chunks_per_page=args.chunks_per_page)
    es.generate_embeddings(batch_size=2)
    es.create_faiss_index()
    es.save()

    qa = QASystem(es)
    qa.load_models()
    sent = SentimentAnalyzer()
    summ = Summarizer()

    log("\n✅ Pipeline setup complete.")

if __name__ == "__main__":
    main()