import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import json
import pickle
import warnings
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Paper Intelligence System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme Toggle ──────────────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = []
if 'history' not in st.session_state:
    st.session_state.history = []

def get_css(dark):
    bg      = "#0e1117" if dark else "#f8f9fa"
    card    = "#1e2130" if dark else "#ffffff"
    text    = "#fafafa" if dark else "#111111"
    border  = "#444"    if dark else "#dee2e6"
    accent  = "#667eea"
    return f"""
<style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    .main-title {{
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }}
    .paper-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid {accent};
    }}
    .score-badge {{
        background: #667eea22;
        border: 1px solid #667eea;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: bold;
        color: #667eea;
    }}
    .keyword-chip {{
        background: #2ecc7122;
        border: 1px solid #2ecc71;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #2ecc71;
        margin-right: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
"""

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ─── Load Models & Data ────────────────────────────────────────
@st.cache_resource
def load_models():
    model      = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', device=-1)
    kw_model   = KeyBERT()
    return model, summarizer, kw_model

@st.cache_resource
def load_data():
    index      = faiss.read_index('outputs/faiss_index/papers.index')
    df         = pd.read_pickle('outputs/data/papers_df.pkl')
    embeddings = np.load('outputs/data/embeddings.npy')
    return index, df, embeddings

# ─── Core Functions ────────────────────────────────────────────
def semantic_search(query, top_k=10, mmr=True, diversity=0.3):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    fetch_k = top_k * 3 if mmr else top_k
    scores, indices = index.search(q_emb, fetch_k)

    results = [{'idx': int(idx), 'title': df['title'].iloc[idx],
                 'abstract': df['abstract'].iloc[idx], 'score': float(score)}
               for idx, score in zip(indices[0], scores[0])]

    if not mmr:
        return results[:top_k]

    candidate_embeddings = np.array([embeddings[r['idx']] for r in results])
    selected = [0]
    while len(selected) < top_k:
        remaining = [i for i in range(len(results)) if i not in selected]
        if not remaining: break
        mmr_scores = []
        for i in remaining:
            relevance = results[i]['score']
            sim_sel = max(cosine_similarity(
                candidate_embeddings[i].reshape(1,-1),
                candidate_embeddings[j].reshape(1,-1)
            )[0][0] for j in selected)
            mmr_scores.append((1-diversity)*relevance - diversity*sim_sel)
        selected.append(remaining[np.argmax(mmr_scores)])
    return [results[i] for i in selected]

def summarize_text(text):
    try:
        text = text[:1024]
        out = summarizer(text, max_length=120, min_length=40, do_sample=False, truncation=True)
        return out[0]['summary_text']
    except:
        return text[:300] + '...'

def get_keywords(text, top_n=8):
    kws = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2),
                                     stop_words='english', use_mmr=True,
                                     diversity=0.5, top_n=top_n)
    return [k[0] for k in kws]

def add_to_history(query, num_results):
    st.session_state.history.insert(0, {
        'query': query, 'results': num_results,
        'timestamp': datetime.now().strftime('%d %b %Y, %H:%M')
    })
    st.session_state.history = st.session_state.history[:30]

def add_bookmark(paper):
    if not any(b['title'] == paper['title'] for b in st.session_state.bookmarks):
        st.session_state.bookmarks.append({
            'title': paper['title'],
            'abstract': paper['abstract'][:300],
            'score': paper['score'],
            'saved_at': datetime.now().strftime('%d %b %Y')
        })
        return True
    return False

def export_pdf(query, results, summaries, keywords_list):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=1*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle('T', parent=styles['Title'], fontSize=16,
                                    textColor=colors.HexColor('#1F4E79'))
    heading_style = ParagraphStyle('H', parent=styles['Heading2'], fontSize=11,
                                    textColor=colors.HexColor('#2E75B6'))
    body_style    = ParagraphStyle('B', parent=styles['Normal'], fontSize=9, leading=13)
    meta_style    = ParagraphStyle('M', parent=styles['Normal'], fontSize=8,
                                    textColor=colors.gray)
    story = [
        Paragraph('AI Research Paper Intelligence System', title_style),
        Paragraph('CBSOT Summer Internship 2026', meta_style),
        Spacer(1, 0.1*inch),
        HRFlowable(width='100%', thickness=2, color=colors.HexColor('#2E75B6')),
        Spacer(1, 0.1*inch),
        Paragraph(f'Query: <b>{query}</b>', body_style),
        Paragraph(f'Papers: {len(results)} | Generated: {datetime.now().strftime("%d %B %Y")}', meta_style),
        Spacer(1, 0.15*inch),
    ]
    for i, r in enumerate(results):
        story += [
            Paragraph(f'{i+1}. {r["title"]}', heading_style),
            Paragraph(f'Score: {r["score"]:.4f}', meta_style),
        ]
        if i < len(summaries):
            story.append(Paragraph(f'<b>Summary:</b> {summaries[i]}', body_style))
        if i < len(keywords_list):
            story.append(Paragraph(f'<b>Keywords:</b> {", ".join(keywords_list[i])}', body_style))
        abstract_short = r['abstract'][:350] + '...' if len(r['abstract']) > 350 else r['abstract']
        story += [
            Paragraph(f'<b>Abstract:</b> {abstract_short}', body_style),
            HRFlowable(width='100%', thickness=0.5, color=colors.lightgrey),
            Spacer(1, 0.08*inch),
        ]
    doc.build(story)
    buffer.seek(0)
    return buffer

def plot_similarity_graph(results):
    paper_embeddings = np.array([embeddings[r['idx']] for r in results])
    sim_matrix = cosine_similarity(paper_embeddings)
    G = nx.Graph()
    for i, r in enumerate(results):
        short = r['title'][:35] + '...' if len(r['title']) > 35 else r['title']
        G.add_node(i, title=short, score=r['score'])
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            if sim_matrix[i][j] > 0.72:
                G.add_edge(i, j, weight=float(sim_matrix[i][j]))

    fig, ax = plt.subplots(figsize=(10, 7))
    bg = '#0e1117' if st.session_state.dark_mode else '#ffffff'
    text_c = 'white' if st.session_state.dark_mode else 'black'
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    pos = nx.spring_layout(G, seed=42, k=2)
    scores = [G.nodes[n]['score'] for n in G.nodes]
    node_colors = plt.cm.YlOrRd(np.array(scores) / max(scores) if max(scores) > 0 else scores)
    edge_weights = [G[u][v]['weight']*3 for u,v in G.edges] or [1]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='#667eea', ax=ax)
    labels = {n: G.nodes[n]['title'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color=text_c, ax=ax)
    edge_labels = {(u,v): f"{G[u][v]['weight']:.2f}" for u,v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)

    ax.set_title('Paper Similarity Graph\n(Edge = cosine similarity > 0.72)', color=text_c, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_citation_network(queries, results_per_topic):
    colors_list = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']
    all_papers  = []
    for topic_idx, (query, results) in enumerate(zip(queries, results_per_topic)):
        for r in results:
            r['topic'] = query; r['topic_idx'] = topic_idx
            all_papers.append(r)

    G = nx.Graph()
    for i, p in enumerate(all_papers):
        short = p['title'][:30] + '...' if len(p['title']) > 30 else p['title']
        G.add_node(i, title=short, color=colors_list[p['topic_idx'] % len(colors_list)])

    paper_embeddings = np.array([embeddings[p['idx']] for p in all_papers])
    sim_matrix = cosine_similarity(paper_embeddings)

    for i in range(len(all_papers)):
        for j in range(i+1, len(all_papers)):
            same = all_papers[i]['topic_idx'] == all_papers[j]['topic_idx']
            threshold = 0.70 if same else 0.82
            if sim_matrix[i][j] > threshold:
                G.add_edge(i, j, weight=float(sim_matrix[i][j]), cross=not same)

    bg = '#0e1117' if st.session_state.dark_mode else '#ffffff'
    text_c = 'white' if st.session_state.dark_mode else 'black'
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    pos = nx.spring_layout(G, seed=42, k=1.5)
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    intra = [(u,v) for u,v,d in G.edges(data=True) if not d.get('cross')]
    cross = [(u,v) for u,v,d in G.edges(data=True) if d.get('cross')]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=intra, width=1.5, alpha=0.4, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cross, width=2.5, alpha=0.8,
                           edge_color='white' if st.session_state.dark_mode else 'black',
                           style='dashed', ax=ax)
    labels = {n: G.nodes[n]['title'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color=text_c, ax=ax)

    from matplotlib.patches import Patch
    legend = [Patch(color=colors_list[i%len(colors_list)], label=q[:25]) for i,q in enumerate(queries)]
    ax.legend(handles=legend, loc='upper left', fontsize=8, facecolor=bg,
              labelcolor=text_c, title='Topics', title_fontsize=8)
    ax.set_title(f'Citation Network — {len(queries)} Topics\n(Dashed = Cross-topic links)',
                 color=text_c, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig

# ─── LOAD ──────────────────────────────────────────────────────
try:
    with st.spinner("Loading models & index (first load ~60s)..."):
        model, summarizer, kw_model = load_models()
        index, df, embeddings = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Could not load index. Run Notebook 1 first!\n\nError: {e}")

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 Research Paper Intelligence")
    st.markdown("*CBSOT Summer Internship 2026*")
    st.divider()

    page = st.radio("Navigate", [
        "🔍 Search Papers",
        "🌐 Similarity Graph",
        "🕸️ Citation Network",
        "⚖️ Compare Papers",
        "🔖 Bookmarks",
        "🕐 Search History"
    ])
    st.divider()

    theme_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.divider()
    st.caption(f"📌 Bookmarks: {len(st.session_state.bookmarks)}")
    st.caption(f"🕐 Searches: {len(st.session_state.history)}")
    st.caption("Dataset: 50k ArXiv ML Papers")
    st.caption("Model: all-MiniLM-L6-v2 + FAISS")

# ─── HEADER ────────────────────────────────────────────────────
st.markdown('<div class="main-title">📚 AI Research Paper Intelligence System</div>', unsafe_allow_html=True)
st.markdown("Semantic Search • BART Summarization • KeyBERT • Similarity Graph • Citation Network • Compare • PDF Export")
st.divider()

if not data_loaded:
    st.stop()

# ═══════════════════════════════════════════════════
# PAGE 1: Search
# ═══════════════════════════════════════════════════
if page == "🔍 Search Papers":
    st.header("🔍 Semantic Search")

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        query = st.text_input("Search query", placeholder="e.g. attention mechanism transformer NLP",
                              label_visibility="collapsed")
    with col2:
        top_k = st.selectbox("Results", [5, 10, 15, 20], index=1)
    with col3:
        use_mmr = st.checkbox("MMR", value=True, help="Maximal Marginal Relevance — diverse results")

    summarize_all = st.checkbox("Auto-summarize all results (slower)", value=False)

    if query:
        with st.spinner(f"Searching {index.ntotal:,} papers..."):
            results = semantic_search(query, top_k=top_k, mmr=use_mmr)
            add_to_history(query, len(results))

        st.success(f"Found **{len(results)}** papers for: *{query}*")
        if use_mmr:
            st.caption("✨ MMR diversity re-ranking applied")

        # Export button
        if st.button("📄 Export All Results as PDF"):
            with st.spinner("Generating PDF..."):
                summaries = [summarize_text(r['abstract']) for r in results]
                kws_list  = [get_keywords(r['abstract']) for r in results]
                pdf_buffer = export_pdf(query, results, summaries, kws_list)
            st.download_button(
                "⬇️ Download PDF",
                data=pdf_buffer,
                file_name=f"results_{query[:30].replace(' ','_')}.pdf",
                mime="application/pdf"
            )

        st.divider()

        for i, r in enumerate(results):
            with st.container():
                col1, col2 = st.columns([8, 2])
                with col1:
                    st.markdown(f"**{i+1}. {r['title']}**")
                with col2:
                    st.markdown(f'<span class="score-badge">Score: {r["score"]:.4f}</span>',
                                unsafe_allow_html=True)

                tab1, tab2, tab3 = st.tabs(["📄 Abstract", "✂️ Summary", "🏷️ Keywords"])

                with tab1:
                    st.write(r['abstract'])

                with tab2:
                    if st.button("Generate Summary", key=f"sum_{i}") or summarize_all:
                        with st.spinner("Summarizing..."):
                            summary = summarize_text(r['abstract'])
                        st.info(summary)

                with tab3:
                    if st.button("Extract Keywords", key=f"kw_{i}"):
                        with st.spinner("Extracting..."):
                            kws = get_keywords(r['abstract'])
                        kw_html = " ".join([f'<span class="keyword-chip">{k}</span>' for k in kws])
                        st.markdown(kw_html, unsafe_allow_html=True)

                col_bm, _ = st.columns([2, 8])
                with col_bm:
                    if st.button("🔖 Bookmark", key=f"bm_{i}"):
                        if add_bookmark(r):
                            st.success("Bookmarked!")
                        else:
                            st.warning("Already saved!")

                st.markdown("---")

# ═══════════════════════════════════════════════════
# PAGE 2: Similarity Graph
# ═══════════════════════════════════════════════════
elif page == "🌐 Similarity Graph":
    st.header("🌐 Paper Similarity Graph")
    st.markdown("Visualize how semantically similar papers connect to each other. **Nodes = papers, Edges = cosine similarity.**")

    col1, col2 = st.columns([3, 1])
    with col1:
        sg_query = st.text_input("Query for graph", placeholder="e.g. graph neural networks")
    with col2:
        sg_k = st.slider("Papers", 5, 15, 8)

    if sg_query and st.button("🌐 Build Graph"):
        with st.spinner("Building similarity graph..."):
            results = semantic_search(sg_query, top_k=sg_k, mmr=False)
            fig = plot_similarity_graph(results)
        st.pyplot(fig)

        st.subheader("Papers in Graph")
        for i, r in enumerate(results):
            st.markdown(f"**{i+1}.** {r['title']} — Score: `{r['score']:.4f}`")

# ═══════════════════════════════════════════════════
# PAGE 3: Citation Network
# ═══════════════════════════════════════════════════
elif page == "🕸️ Citation Network":
    st.header("🕸️ Citation Network")
    st.markdown("Enter **2-4 different topics** to see how papers across topics connect.")

    num_topics = st.slider("Number of topics", 2, 4, 3)
    topics = []
    for i in range(num_topics):
        t = st.text_input(f"Topic {i+1}", key=f"topic_{i}",
                          placeholder=f"e.g. {'transformer attention' if i==0 else 'GAN image synthesis' if i==1 else 'reinforcement learning'}")
        topics.append(t)

    papers_per = st.slider("Papers per topic", 3, 8, 4)

    if all(topics) and st.button("🕸️ Build Citation Network"):
        with st.spinner("Building network..."):
            all_results = [semantic_search(t, top_k=papers_per, mmr=True) for t in topics]
            fig = plot_citation_network(topics, all_results)
        st.pyplot(fig)
        st.caption("Solid lines = same-topic similarity | Dashed = cross-topic connections")

# ═══════════════════════════════════════════════════
# PAGE 4: Compare Papers
# ═══════════════════════════════════════════════════
elif page == "⚖️ Compare Papers":
    st.header("⚖️ Compare Two Papers")
    st.markdown("Find the best paper for two different queries and compare them side-by-side.")

    col1, col2 = st.columns(2)
    with col1:
        q1 = st.text_input("Query A", placeholder="e.g. BERT language model")
    with col2:
        q2 = st.text_input("Query B", placeholder="e.g. GPT generative model")

    if q1 and q2 and st.button("⚖️ Compare"):
        with st.spinner("Finding and comparing papers..."):
            r1 = semantic_search(q1, top_k=1, mmr=False)[0]
            r2 = semantic_search(q2, top_k=1, mmr=False)[0]

            emb1 = embeddings[r1['idx']].reshape(1,-1)
            emb2 = embeddings[r2['idx']].reshape(1,-1)
            cross_sim = cosine_similarity(emb1, emb2)[0][0]

            sum1 = summarize_text(r1['abstract'])
            sum2 = summarize_text(r2['abstract'])
            kw1  = get_keywords(r1['abstract'])
            kw2  = get_keywords(r2['abstract'])
            common = set(kw1) & set(kw2)

        st.metric("Cross-Paper Similarity", f"{cross_sim:.4f}",
                  help="How similar these two papers are to each other")
        if common:
            st.info(f"**Common Keywords:** {', '.join(common)}")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### 📄 Paper A")
            st.markdown(f"**{r1['title']}**")
            st.caption(f"Relevance: {r1['score']:.4f}")
            st.markdown("**Summary:**")
            st.info(sum1)
            st.markdown("**Keywords:**")
            kw_html = " ".join([f'<span class="keyword-chip">{k}</span>' for k in kw1])
            st.markdown(kw_html, unsafe_allow_html=True)
            if st.button("🔖 Bookmark A"):
                add_bookmark(r1); st.success("Saved!")

        with col2:
            st.markdown(f"### 📄 Paper B")
            st.markdown(f"**{r2['title']}**")
            st.caption(f"Relevance: {r2['score']:.4f}")
            st.markdown("**Summary:**")
            st.info(sum2)
            st.markdown("**Keywords:**")
            kw_html = " ".join([f'<span class="keyword-chip">{k}</span>' for k in kw2])
            st.markdown(kw_html, unsafe_allow_html=True)
            if st.button("🔖 Bookmark B"):
                add_bookmark(r2); st.success("Saved!")

        # Keyword overlap chart
        st.divider()
        st.subheader("Keyword Comparison")
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        bg = '#0e1117' if st.session_state.dark_mode else '#ffffff'
        text_c = 'white' if st.session_state.dark_mode else 'black'
        for ax, kws, color, title in [
            (axes[0], kw1, '#3498db', r1['title'][:40]),
            (axes[1], kw2, '#e74c3c', r2['title'][:40])
        ]:
            fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
            ax.barh(range(len(kws)), [1]*len(kws), color=color, alpha=0.8)
            ax.set_yticks(range(len(kws)))
            ax.set_yticklabels(kws, fontsize=9, color=text_c)
            ax.set_title(title, fontsize=9, fontweight='bold', color=text_c)
            ax.set_xticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

# ═══════════════════════════════════════════════════
# PAGE 5: Bookmarks
# ═══════════════════════════════════════════════════
elif page == "🔖 Bookmarks":
    st.header("🔖 Saved Papers")

    if not st.session_state.bookmarks:
        st.info("No bookmarks yet! Search papers and click 🔖 Bookmark to save them here.")
    else:
        st.success(f"{len(st.session_state.bookmarks)} papers saved")

        if st.button("🗑️ Clear All Bookmarks"):
            st.session_state.bookmarks = []
            st.rerun()

        # Export bookmarks as PDF
        if st.button("📄 Export Bookmarks as PDF"):
            bm_results = [{'title': b['title'], 'abstract': b['abstract'],
                           'score': b['score'], 'idx': 0} for b in st.session_state.bookmarks]
            pdf_buf = export_pdf("My Bookmarks", bm_results, [], [])
            st.download_button("⬇️ Download PDF", data=pdf_buf,
                               file_name="bookmarks.pdf", mime="application/pdf")

        st.divider()
        for i, bm in enumerate(st.session_state.bookmarks):
            with st.expander(f"📄 {bm['title'][:80]}"):
                st.caption(f"Saved: {bm['saved_at']} | Score: {bm['score']:.4f}")
                st.write(bm['abstract'])
                if st.button("🗑️ Remove", key=f"rm_{i}"):
                    st.session_state.bookmarks.pop(i)
                    st.rerun()

# ═══════════════════════════════════════════════════
# PAGE 6: Search History
# ═══════════════════════════════════════════════════
elif page == "🕐 Search History":
    st.header("🕐 Search History")

    if not st.session_state.history:
        st.info("No search history yet! Start searching to see your history here.")
    else:
        st.success(f"{len(st.session_state.history)} searches recorded")

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

        for h in st.session_state.history:
            col1, col2, col3 = st.columns([5, 2, 1])
            with col1:
                if st.button(f"🔍 {h['query']}", key=f"hist_{h['timestamp']}",
                             use_container_width=True):
                    st.session_state['prefill_query'] = h['query']
            with col2:
                st.caption(f"{h['results']} results")
            with col3:
                st.caption(h['timestamp'][:6])
