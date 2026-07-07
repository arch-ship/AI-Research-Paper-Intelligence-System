# 📚 AI Research Paper Intelligence System
### CBSOT Summer Internship 2026 | Project 2

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b?logo=streamlit)](https://YOUR-APP-URL.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3-orange)](https://console.groq.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Live Demo →** [YOUR-APP-URL.streamlit.app](https://YOUR-APP-URL.streamlit.app)

| Notebook | Open in Colab |
|---|---|
| 📓 01 — Data Loading & FAISS Index | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10oE40Ek0veA_lfVzTDDCNCReUbPP5fR_?usp=sharing) |
| 📓 02 — Features & Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tkJX3--U0sSulhdFJ_qnv0P1yU9FiLn9?usp=sharing) |

---

## 📌 Project Overview

Research paper discovery is a challenge — with millions of papers published every year, finding **semantically relevant** work is nearly impossible with keyword search alone.

This project builds a complete **AI-powered Research Paper Intelligence System** that:

1. **Searches** 50,000 ArXiv ML papers semantically (meaning-based, not keyword-based)
2. **Compares** Semantic Search vs TF-IDF Search side-by-side
3. **Summarizes** papers using BART (local) or Groq LLaMA3 (AI-powered)
4. **Extracts keywords** using KeyBERT vs TF-IDF comparison
5. **Visualizes** paper similarity as an interactive NetworkX graph
6. **Maps** topic clusters via Citation Network
7. **Compares** two papers side-by-side with Groq AI analysis
8. **Exports** results as a formatted PDF report
9. **Saves** papers to Bookmarks and tracks Search History

---

## 🤖 Groq API Key Setup

This project uses **Groq LLaMA3-8b** for AI-powered summarization, search insights, and paper comparison analysis.

**Get free API key (takes 2 min):**
1. Go to 👉 [console.groq.com](https://console.groq.com)
2. Sign up (free) → API Keys → Create Key
3. Copy the key (starts with `gsk_...`)

**Where to add:**

| Where | How |
|---|---|
| **Streamlit App** | Sidebar → "🤖 Groq API Key" field → paste key |
| **Notebook 2** | Cell 9 → `GROQ_API_KEY = "your_groq_api_key_here"` → replace |

> Core features (search, BART summary, graphs, PDF) work without Groq. Groq powers the 🤖 AI analysis buttons only.

---

## 🗂️ Repository Structure

```
AI-Research-Paper-Intelligence-System/
│
├── 📓 notebook/
│   ├── 01_Data_and_Index.ipynb        # Dataset loading, EDA, FAISS index
│   └── 02_App_and_Features.ipynb      # All features tested end-to-end
│
├── 🌐 app/
│   └── app.py                         # Streamlit dashboard (7 pages)
│
├── 📊 outputs/
│   ├── faiss_index/papers.index       # FAISS index (auto-generated)
│   ├── data/papers_df.pkl             # Processed dataframe
│   ├── data/embeddings.npy            # Paper embeddings
│   ├── 01_text_length_distribution.png
│   ├── 02_top_keywords.png
│   ├── 03_similarity_graph.png
│   ├── 04_citation_network.png
│   ├── 05_paper_comparison.png
│   ├── 06_semantic_vs_tfidf.png
│   └── 07_keybert_vs_tfidf.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | `CShorten/ML-ArXiv-Papers` (HuggingFace) |
| **Total Papers** | 50,000 ArXiv ML papers |
| **Fields** | `title`, `abstract` |
| **Domain** | Machine Learning, AI, Deep Learning |
| **API Key Required** | ❌ None — fully public dataset |

---

## 🔧 Pipeline

### Step 1 — Data Loading & EDA
- Load `CShorten/ML-ArXiv-Papers` from HuggingFace datasets
- Sample 50,000 papers, clean and combine title + abstract
- EDA: Abstract/title length distributions, top keyword frequency

### Step 2 — Semantic Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Encode all 50,000 papers → 384-dimensional dense vectors
- L2 normalization for cosine similarity via dot product

### Step 3 — FAISS Index
- `faiss.IndexFlatIP` (Inner Product = cosine after normalization)
- All 50,000 vectors indexed for millisecond retrieval

### Step 4 — TF-IDF Index
- `sklearn.TfidfVectorizer` with `max_features=10000`, `ngram_range=(1,2)`
- Sparse keyword-based search for comparison with semantic search

### Step 5 — MMR Search
- Query → embedding → FAISS top-K candidates
- **MMR (Maximal Marginal Relevance)** re-ranking for diverse results

### Step 6 — Summarization (2 options)
- **BART:** `sshleifer/distilbart-cnn-12-6` — local, no API key
- **Groq LLaMA3:** `llama3-8b-8192` — faster, AI-powered, needs free key

### Step 7 — Keyword Extraction (2 methods compared)
- **KeyBERT** with MMR diversity (`keyphrase_ngram_range=(1,2)`)
- **TF-IDF** from the same vectorizer — direct comparison

---

## 🌟 Unique Features

### 📊 Semantic vs TF-IDF Search Comparison
Compare dense semantic search vs sparse keyword search side-by-side. See which papers each method finds and how many overlap.

![Semantic vs TF-IDF](outputs/06_semantic_vs_tfidf.png)

### 🏷️ KeyBERT vs TF-IDF Keyword Comparison
Extract keywords using both methods and compare — semantic understanding (KeyBERT) vs term frequency (TF-IDF).

![KeyBERT vs TF-IDF](outputs/07_keybert_vs_tfidf.png)

### 🌐 Paper Similarity Graph
Visualize semantic similarity between retrieved papers as a **NetworkX graph**.
- Nodes = papers (colored by relevance score)
- Edges = cosine similarity > threshold with exact scores

![Similarity Graph](outputs/03_similarity_graph.png)

### 🕸️ Citation Network
Enter 2–4 different topics to see how papers across topics interconnect.
- Each topic = a colored cluster
- Dashed edges = cross-topic semantic overlap

![Citation Network](outputs/04_citation_network.png)

### ⚖️ Compare Two Papers (Semantic + TF-IDF + Groq)
Find best paper for two queries and compare:
- BART summaries side-by-side
- Semantic similarity score
- TF-IDF similarity score
- Groq LLaMA AI analysis of key differences

![Paper Comparison](outputs/05_paper_comparison.png)

### 📄 PDF Export
Export search results as formatted PDF (ReportLab) — query, scores, summaries, keywords, abstracts.

### 🔖 Bookmarks + 🕐 Search History
Save interesting papers and track all past queries with one-click re-search.

---

## 🛠️ Tech Stack

| Tool | Version | Use |
|---|---|---|
| Python | 3.10 | Core language |
| `sentence-transformers` | ≥2.7.0 | `all-MiniLM-L6-v2` embeddings |
| FAISS | ≥1.8.0 | Dense vector similarity search |
| HuggingFace Transformers | ≥4.40.0 | DistilBART summarization |
| KeyBERT | ≥0.8.0 | Keyword extraction with MMR |
| Scikit-Learn TF-IDF | ≥1.5.0 | Sparse keyword search + comparison |
| Groq | latest | LLaMA3-8b AI analysis |
| NetworkX | ≥3.3 | Similarity & citation graphs |
| ReportLab | ≥4.1.0 | PDF export |
| Streamlit | ≥1.35.0 | Interactive dashboard |
| Pandas / NumPy | latest | Data processing |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
Click the Colab badges above — no local setup needed!

### Option 2: Local
```bash
git clone https://github.com/arch-ship/AI-Research-Paper-Intelligence-System.git
cd AI-Research-Paper-Intelligence-System
pip install -r requirements.txt

# Step 1: Build FAISS index
jupyter notebook notebook/01_Data_and_Index.ipynb

# Step 2: Test all features
jupyter notebook notebook/02_App_and_Features.ipynb

# Step 3: Launch app
streamlit run app/app.py
```

### Option 3: Live Demo
👉 [YOUR-APP-URL.streamlit.app](https://YOUR-APP-URL.streamlit.app)

> No API keys required for core features. Groq key optional for AI analysis.

---

## 📱 Streamlit App — 7 Pages

| Page | Feature |
|---|---|
| 🔍 Search Papers | Semantic search, BART + Groq summaries, KeyBERT + TF-IDF keywords, PDF export |
| 📊 Semantic vs TF-IDF | Side-by-side search method comparison with bar charts |
| 🌐 Similarity Graph | NetworkX graph of paper similarities |
| 🕸️ Citation Network | Multi-topic cluster visualization |
| ⚖️ Compare Papers | Side-by-side comparison + Semantic sim + TF-IDF sim + Groq analysis |
| 🔖 Bookmarks | Save and export papers as PDF |
| 🕐 Search History | Track all past queries |

Dark/Light theme toggle available in sidebar.

---

## 📁 Output Files

| File | Description |
|---|---|
| `outputs/01_text_length_distribution.png` | Abstract & title length distributions |
| `outputs/02_top_keywords.png` | Top 20 keywords in paper titles |
| `outputs/03_similarity_graph.png` | Paper similarity network |
| `outputs/04_citation_network.png` | Multi-topic citation network |
| `outputs/05_paper_comparison.png` | Side-by-side keyword comparison |
| `outputs/06_semantic_vs_tfidf.png` | Semantic vs TF-IDF search results |
| `outputs/07_keybert_vs_tfidf.png` | KeyBERT vs TF-IDF keywords |
| `outputs/search_results.pdf` | Exported search results |
| `outputs/bookmarks.json` | Saved bookmarks |
| `outputs/search_history.json` | Search history |

---

## 🎓 Acknowledgements

- **Mentor:** Aryesh Rai Sir — for guidance throughout this internship
- **CBSOT Team:** Kartik Mathur Sir, Varun Kohli Sir, Monu Kumar Sir
- **Organization:** Coding Blocks School of Technology
- **Dataset:** `CShorten/ML-ArXiv-Papers` via HuggingFace

---

## 🔮 Future Enhancements

- [ ] LangChain RetrievalQA for conversational paper Q&A
- [ ] GLiNER zero-shot NER for tech entity extraction
- [ ] Paper recommendation based on reading history
- [ ] Multi-language abstract support

---

*Built with ❤️ during CBSOT Summer Internship 2026*