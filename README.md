# 📚 AI Research Paper Intelligence System
### CBSOT Summer Internship 2026 | Project 2

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b?logo=streamlit)](https://YOUR-APP-URL.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue)](https://github.com/facebookresearch/faiss)
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
2. **Summarizes** papers automatically using BART
3. **Extracts keywords** using KeyBERT with MMR diversity
4. **Visualizes** paper similarity as an interactive graph
5. **Maps** topic clusters via Citation Network
6. **Compares** two papers side-by-side
7. **Exports** results as a formatted PDF report
8. **Saves** papers to Bookmarks and tracks Search History

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
│   └── app.py                         # Streamlit dashboard (6 pages)
│
├── 📊 outputs/
│   ├── faiss_index/papers.index       # FAISS index (auto-generated)
│   ├── data/papers_df.pkl             # Processed dataframe
│   ├── data/embeddings.npy            # Paper embeddings
│   ├── 01_text_length_distribution.png
│   ├── 02_top_keywords.png
│   ├── 03_similarity_graph.png
│   ├── 04_citation_network.png
│   └── 05_paper_comparison.png
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
| **API Key Required** | ❌ None — fully public |

---

## 🔧 Pipeline

### Step 1 — Data Loading & EDA
- Load `CShorten/ML-ArXiv-Papers` from HuggingFace datasets
- Sample 50,000 papers, clean and combine title + abstract
- EDA: Abstract/title length distributions, top keyword frequency analysis

### Step 2 — Semantic Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Encode all 50,000 papers → 384-dimensional dense vectors
- L2 normalization for cosine similarity via dot product

### Step 3 — FAISS Index
- `faiss.IndexFlatIP` (Inner Product = cosine after normalization)
- All 50,000 vectors indexed for millisecond search

### Step 4 — MMR Search
- Query → embedding → FAISS top-K candidates
- **MMR (Maximal Marginal Relevance)** re-ranking for diverse results
- Balances relevance vs redundancy via `diversity` parameter

### Step 5 — Summarization
- Model: `sshleifer/distilbart-cnn-12-6`
- Summarizes abstracts to 40–120 word summaries

### Step 6 — Keyword Extraction
- KeyBERT with MMR diversity (`keyphrase_ngram_range=(1,2)`)
- Returns top-8 diverse keyphrases per paper

---

## 🌟 Unique Features

### 🌐 Paper Similarity Graph
Visualize semantic similarity between retrieved papers as a **NetworkX graph**.
- Nodes = papers (colored by relevance score)
- Edges = cosine similarity > threshold
- Edge labels show exact similarity scores

![Similarity Graph](outputs/03_similarity_graph.png)

### 🕸️ Citation Network
Enter **2–4 different topics** to see how papers across topics interconnect.
- Each topic = a colored cluster
- Solid edges = intra-topic similarity
- Dashed edges = cross-topic semantic overlap

![Citation Network](outputs/04_citation_network.png)

### ⚖️ Compare Two Papers Side-by-Side
Find the best paper for two different queries and compare:
- Summaries side-by-side
- Keyword comparison chart
- Cross-paper similarity score
- Common keywords highlighted

![Paper Comparison](outputs/05_paper_comparison.png)

### 📄 PDF Export
Export any search results as a **formatted PDF report** (ReportLab):
- Query, timestamp, paper count
- Per-paper: title, relevance score, summary, keywords, abstract

### 🔖 Bookmarks
Save interesting papers during search sessions:
- Persistent via JSON file
- Remove individual bookmarks
- Export all bookmarks as PDF

### 🕐 Search History
Track all queries automatically:
- Last 30 searches stored
- One-click re-search from history
- Timestamp per query

---

## 🛠️ Tech Stack

| Tool | Version | Use |
|---|---|---|
| Python | 3.10 | Core language |
| `sentence-transformers` | ≥2.7.0 | `all-MiniLM-L6-v2` embeddings |
| FAISS | ≥1.8.0 | Vector similarity search |
| HuggingFace Transformers | ≥4.40.0 | DistilBART summarization |
| KeyBERT | ≥0.8.0 | Keyword extraction with MMR |
| NetworkX | ≥3.3 | Similarity & citation graphs |
| ReportLab | ≥4.1.0 | PDF export |
| Scikit-Learn | ≥1.5.0 | Cosine similarity |
| Streamlit | ≥1.35.0 | Interactive dashboard |
| Pandas / NumPy | latest | Data processing |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
Click the Colab badges above — no setup needed!

### Option 2: Local
```bash
git clone https://github.com/arch-ship/AI-Research-Paper-Intelligence-System.git
cd AI-Research-Paper-Intelligence-System
pip install -r requirements.txt

# Step 1: Run Notebook 1 to build FAISS index
jupyter notebook notebook/01_Data_and_Index.ipynb

# Step 2: Run Notebook 2 to test all features
jupyter notebook notebook/02_App_and_Features.ipynb

# Step 3: Launch Streamlit app
streamlit run app/app.py
```

### Option 3: Live Demo
👉 [YOUR-APP-URL.streamlit.app](https://YOUR-APP-URL.streamlit.app)

> **Note:** No API keys required. All models download automatically from HuggingFace on first run.

---

## 📱 Streamlit App — 6 Pages

| Page | Feature |
|---|---|
| 🔍 Search Papers | Semantic search with MMR toggle, auto-summarize, PDF export |
| 🌐 Similarity Graph | NetworkX graph of paper similarities |
| 🕸️ Citation Network | Multi-topic cluster visualization |
| ⚖️ Compare Papers | Side-by-side paper comparison |
| 🔖 Bookmarks | Save and export papers |
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
- [ ] Multi-language support

---

*Built with ❤️ during CBSOT Summer Internship 2026*
