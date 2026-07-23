"""
AI Research Paper Intelligence System
CBSOT Summer Internship 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import json
import warnings
import networkx as nx
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from groq import Groq
from dotenv import load_dotenv
import io

load_dotenv()
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State ──────────────────────────────────────────────
defaults = {
    'dark_mode' : True,
    'bookmarks' : [],
    'history'   : [],
    'groq_key'  : os.getenv("GROQ_API_KEY", ""),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ────────────────────────────────────────────────────────
def get_css(dark):
    bg     = "#0d1117" if dark else "#f6f8fa"
    card   = "#161b22" if dark else "#ffffff"
    card2  = "#21262d" if dark else "#f0f4f8"
    text   = "#e6edf3" if dark else "#24292f"
    muted  = "#8b949e" if dark else "#656d76"
    border = "#30363d" if dark else "#d0d7de"
    accent = "#58a6ff"
    green  = "#3fb950"
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background-color:{bg}; color:{text}; }}
section[data-testid="stSidebar"] {{
    background-color:{card} !important;
    border-right:1px solid {border};
}}
.main-title {{
    font-size:1.8rem; font-weight:700; color:{accent};
    letter-spacing:-0.5px; margin-bottom:2px;
}}
.subtitle {{
    font-size:0.82rem; color:{muted}; margin-bottom:20px;
}}
.paper-card {{
    background:{card}; border:1px solid {border};
    border-radius:8px; padding:16px 20px;
    margin-bottom:10px; border-left:3px solid {accent};
}}
.keyword-chip {{
    display:inline-block; background:{card2};
    border:1px solid {border}; border-radius:4px;
    padding:2px 8px; font-size:0.75rem;
    font-family:'JetBrains Mono',monospace;
    color:{accent}; margin:2px;
}}
.score-badge {{
    background:{card2}; border:1px solid {border};
    border-radius:4px; padding:2px 8px;
    font-size:0.75rem; font-family:'JetBrains Mono',monospace;
    color:{green};
}}
.chat-user {{
    background:{card2}; border:1px solid {border};
    border-radius:8px; padding:12px 16px;
    margin:8px 0; border-left:3px solid {accent};
}}
.chat-ai {{
    background:{card}; border:1px solid {border};
    border-radius:8px; padding:12px 16px;
    margin:8px 0; border-left:3px solid {green};
}}
.chat-label {{
    font-size:0.72rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.8px;
    color:{muted}; margin-bottom:6px;
}}
.stat-card {{
    background:{card}; border:1px solid {border};
    border-radius:8px; padding:16px;
    text-align:center;
}}
.stat-number {{
    font-size:1.5rem; font-weight:700; color:{accent};
    font-family:'JetBrains Mono',monospace;
}}
.stat-label {{
    font-size:0.72rem; color:{muted};
    text-transform:uppercase; letter-spacing:0.8px;
}}
.section-label {{
    font-size:0.72rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.8px;
    color:{muted}; margin-bottom:8px;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap:4px; background:{card2};
    border-radius:8px; padding:4px;
    border:1px solid {border};
}}
.stTabs [data-baseweb="tab"] {{
    font-weight:500; font-size:0.82rem;
    border-radius:6px; padding:5px 14px; color:{muted};
}}
.stTabs [aria-selected="true"] {{
    background:{card} !important;
    color:{text} !important; font-weight:600;
}}
.stButton button {{
    border-radius:6px !important;
    font-weight:500 !important; font-size:0.85rem !important;
}}
div[data-testid="metric-container"] {{
    background:{card}; border:1px solid {border}; border-radius:8px; padding:12px;
}}
hr {{ border-color:{border} !important; opacity:0.4; }}
</style>"""

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer   = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    bart        = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
    kw_model    = KeyBERT()
    return embed_model, tokenizer, bart, kw_model

@st.cache_resource
def load_data():
    idx  = faiss.read_index('outputs/faiss_index/papers.index')
    df   = pd.read_pickle('outputs/data/papers_df.pkl')
    embs = np.load('outputs/data/embeddings.npy')
    return idx, df, embs

@st.cache_resource
def build_tfidf(_df):
    vec = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
    mat = vec.fit_transform(_df['text'].tolist())
    return vec, mat

# ── Core Functions ─────────────────────────────────────────────
def semantic_search(query, top_k=10, mmr=True, diversity=0.3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    fetch_k = top_k*3 if mmr else top_k
    scores, indices = index.search(q_emb, fetch_k)
    results = [{'idx':int(idx),'title':df['title'].iloc[idx],
                'abstract':df['abstract'].iloc[idx],'score':float(score)}
               for idx,score in zip(indices[0],scores[0])]
    if not mmr: return results[:top_k]
    cands = np.array([embeddings[r['idx']] for r in results])
    sel   = [0]
    while len(sel) < top_k:
        rem = [i for i in range(len(results)) if i not in sel]
        if not rem: break
        mmr_s = [(1-diversity)*results[i]['score'] -
                 diversity*max(cosine_similarity(cands[i].reshape(1,-1),
                               cands[j].reshape(1,-1))[0][0] for j in sel)
                 for i in rem]
        sel.append(rem[np.argmax(mmr_s)])
    return [results[i] for i in sel]

def tfidf_search(query, top_k=10):
    q_vec  = tfidf_vec.transform([query])
    scores = cos_sim(q_vec, tfidf_mat).flatten()
    top    = scores.argsort()[::-1][:top_k]
    return [{'idx':int(i),'title':df['title'].iloc[i],
             'abstract':df['abstract'].iloc[i],'score':float(scores[i])} for i in top]

def summarize_bart(text):
    try:
        inp = tokenizer(text[:1024], return_tensors='pt', truncation=True, max_length=1024)
        with torch.no_grad():
            ids = bart_model.generate(inp['input_ids'], max_length=120,
                                      min_length=40, num_beams=4, early_stopping=True)
        return tokenizer.decode(ids[0], skip_special_tokens=True)
    except:
        return text[:300]+'...'

def get_keywords(text, top_n=8):
    kws = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2),
                                     stop_words='english', use_mmr=True,
                                     diversity=0.5, top_n=top_n)
    return [k[0] for k in kws]

def get_tfidf_keywords(text, top_n=8):
    doc_vec = tfidf_vec.transform([text])
    feat    = tfidf_vec.get_feature_names_out()
    sc      = doc_vec.toarray().flatten()
    top     = sc.argsort()[::-1][:top_n]
    return [feat[i] for i in top if sc[i] > 0]

def groq_call(prompt, max_tokens=250):
    if not st.session_state.groq_key:
        return "Add Groq API key in sidebar."
    try:
        client = Groq(api_key=st.session_state.groq_key)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=0.4
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        try:
            client = Groq(api_key=st.session_state.groq_key)
            resp   = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens, temperature=0.4
            )
            return resp.choices[0].message.content.strip()
        except Exception as e2:
            return f"Groq error: {e2}"

def groq_summarize(text):
    return groq_call(f"Summarize this ML research paper in 80 words. Key contribution only.\n\nAbstract: {text[:2000]}")

def groq_insights(query, results):
    titles = '\n'.join([f'{i+1}. {r["title"]}' for i,r in enumerate(results)])
    return groq_call(f'Papers for "{query}":\n{titles}\n\nIn 3 sentences, what is the current research trend?')

def groq_compare(r1, r2, sim):
    return groq_call(
        f"Compare these 2 ML papers in under 120 words.\n\n"
        f"Paper A: {r1['title']}\n{r1['abstract'][:400]}\n\n"
        f"Paper B: {r2['title']}\n{r2['abstract'][:400]}\n\n"
        f"Similarity: {sim:.3f}\n\n"
        f"1. Key difference\n2. What each solves\n3. Which is more novel", max_tokens=300)

def add_to_history(query, n):
    st.session_state.history.insert(0,{
        'query':query,'results':n,'timestamp':datetime.now().strftime('%d %b, %H:%M')})
    st.session_state.history = st.session_state.history[:30]

def add_bookmark(paper):
    if not any(b['title']==paper['title'] for b in st.session_state.bookmarks):
        st.session_state.bookmarks.append({
            'title':paper['title'],'abstract':paper['abstract'][:300],
            'score':paper['score'],'saved_at':datetime.now().strftime('%d %b %Y')})
        return True
    return False

def make_pdf(query, results, summaries, kws_list):
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=letter,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=1*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    ts = ParagraphStyle('T',parent=styles['Title'],fontSize=16,
                         textColor=colors.HexColor('#1F4E79'))
    hs = ParagraphStyle('H',parent=styles['Heading2'],fontSize=11,
                         textColor=colors.HexColor('#2E75B6'))
    bs = ParagraphStyle('B',parent=styles['Normal'],fontSize=9,leading=13)
    ms = ParagraphStyle('M',parent=styles['Normal'],fontSize=8,textColor=colors.gray)
    story = [
        Paragraph('AI Research Paper Intelligence System', ts),
        Paragraph('CBSOT Summer Internship 2026', ms),
        Spacer(1,0.1*inch),
        HRFlowable(width='100%',thickness=2,color=colors.HexColor('#2E75B6')),
        Spacer(1,0.1*inch),
        Paragraph(f'Query: <b>{query}</b>', bs),
        Paragraph(f'{len(results)} papers · {datetime.now().strftime("%d %B %Y")}', ms),
        Spacer(1,0.15*inch),
    ]
    for i,r in enumerate(results):
        story += [Paragraph(f'{i+1}. {r["title"]}', hs),
                  Paragraph(f'Score: {r["score"]:.4f}', ms)]
        if i < len(summaries):
            story.append(Paragraph(f'<b>Summary:</b> {summaries[i]}', bs))
        if i < len(kws_list):
            story.append(Paragraph(f'<b>Keywords:</b> {", ".join(kws_list[i])}', bs))
        short = r['abstract'][:350]+'...' if len(r['abstract'])>350 else r['abstract']
        story += [Paragraph(f'<b>Abstract:</b> {short}', bs),
                  HRFlowable(width='100%',thickness=0.5,color=colors.lightgrey),
                  Spacer(1,0.08*inch)]
    doc.build(story)
    buf.seek(0)
    return buf

def sim_graph_fig(results):
    pe  = np.array([embeddings[r['idx']] for r in results])
    sm  = cosine_similarity(pe)
    G   = nx.Graph()
    for i,r in enumerate(results):
        G.add_node(i, title=r['title'][:32]+'...' if len(r['title'])>32 else r['title'],
                   score=r['score'])
    for i in range(len(results)):
        for j in range(i+1,len(results)):
            if sm[i][j]>0.72: G.add_edge(i,j,weight=float(sm[i][j]))
    dark = st.session_state.dark_mode
    bg   = '#0d1117' if dark else '#f6f8fa'
    tc   = '#e6edf3' if dark else '#24292f'
    fig,ax = plt.subplots(figsize=(10,7))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    pos = nx.spring_layout(G,seed=42,k=2)
    sc  = [G.nodes[n]['score'] for n in G.nodes]
    nc  = plt.cm.YlOrRd(np.array(sc)/max(sc)) if sc else ['red']
    nx.draw_networkx_nodes(G,pos,node_color=nc,node_size=1800,alpha=0.9,ax=ax)
    if G.edges:
        ew = [G[u][v]['weight']*3 for u,v in G.edges]
        nx.draw_networkx_edges(G,pos,width=ew,alpha=0.5,edge_color='#58a6ff',ax=ax)
        nx.draw_networkx_edge_labels(G,pos,
            {(u,v):f"{G[u][v]['weight']:.2f}" for u,v in G.edges},font_size=6,ax=ax)
    nx.draw_networkx_labels(G,pos,{n:G.nodes[n]['title'] for n in G.nodes},
                            font_size=7,font_color=tc,ax=ax)
    ax.set_title('Paper Similarity Graph',color=tc,fontweight='bold',fontsize=12)
    ax.axis('off'); plt.tight_layout()
    return fig

def citation_fig(queries, results_per_topic):
    cl    = ['#f85149','#58a6ff','#3fb950','#d29922','#bc8cff']
    all_p = []
    for ti,(q,res) in enumerate(zip(queries,results_per_topic)):
        for r in res: r['topic_idx']=ti; all_p.append(r)
    G = nx.Graph()
    for i,p in enumerate(all_p):
        G.add_node(i, title=p['title'][:28]+'...' if len(p['title'])>28 else p['title'],
                   color=cl[p['topic_idx']%len(cl)])
    pe  = np.array([embeddings[p['idx']] for p in all_p])
    sm  = cosine_similarity(pe)
    for i in range(len(all_p)):
        for j in range(i+1,len(all_p)):
            same = all_p[i]['topic_idx']==all_p[j]['topic_idx']
            if sm[i][j] > (0.70 if same else 0.82):
                G.add_edge(i,j,weight=float(sm[i][j]),cross=not same)
    dark = st.session_state.dark_mode
    bg   = '#0d1117' if dark else '#f6f8fa'
    tc   = '#e6edf3' if dark else '#24292f'
    fig,ax = plt.subplots(figsize=(12,8))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    pos   = nx.spring_layout(G,seed=42,k=1.5)
    nc    = [G.nodes[n]['color'] for n in G.nodes]
    intra = [(u,v) for u,v,d in G.edges(data=True) if not d.get('cross')]
    cross = [(u,v) for u,v,d in G.edges(data=True) if d.get('cross')]
    nx.draw_networkx_nodes(G,pos,node_color=nc,node_size=1200,alpha=0.85,ax=ax)
    if intra: nx.draw_networkx_edges(G,pos,edgelist=intra,width=1.5,alpha=0.4,edge_color='#30363d',ax=ax)
    if cross: nx.draw_networkx_edges(G,pos,edgelist=cross,width=2.5,alpha=0.8,
                                     edge_color=tc,style='dashed',ax=ax)
    nx.draw_networkx_labels(G,pos,{n:G.nodes[n]['title'] for n in G.nodes},
                            font_size=6,font_color=tc,ax=ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cl[i%len(cl)],label=q[:25]) for i,q in enumerate(queries)],
              loc='upper left',fontsize=8,facecolor=bg,labelcolor=tc)
    ax.set_title('Citation Network · Dashed = Cross-topic',color=tc,fontweight='bold')
    ax.axis('off'); plt.tight_layout()
    return fig

# ── LOAD ──────────────────────────────────────────────────────
try:
    with st.spinner("Loading models and index..."):
        embed_model, tokenizer, bart_model, kw_model = load_models()
        index, df, embeddings = load_data()
        tfidf_vec, tfidf_mat  = build_tfidf(df)
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Run Notebook 1 first to build the FAISS index.\n\nError: {e}")

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Research Paper Intelligence")
    st.markdown("<span style='color:#8b949e;font-size:0.8rem'>CBSOT Summer Internship 2026</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<span style='font-size:0.72rem;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.8px'>GROQ API KEY</span>", unsafe_allow_html=True)
    groq_input = st.text_input("Key", type="password",
                                value=st.session_state.groq_key,
                                placeholder="gsk_... or set in .env",
                                label_visibility="collapsed")
    if groq_input != st.session_state.groq_key:
        st.session_state.groq_key = groq_input
    if st.session_state.groq_key:
        st.markdown('<span style="color:#3fb950;font-size:0.8rem">&#9679; Groq Connected</span>', unsafe_allow_html=True)
    else:
        st.caption("[console.groq.com](https://console.groq.com) · free key")

    st.divider()
    st.caption(f"Bookmarks: {len(st.session_state.bookmarks)}")
    st.caption(f"Searches: {len(st.session_state.history)}")
    st.caption("50,000 ArXiv ML Papers")

    st.divider()
    if st.button("Light Mode" if st.session_state.dark_mode else "Dark Mode", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── HEADER ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">Research Paper Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Semantic Search · TF-IDF · BART · KeyBERT · Groq LLaMA · Graphs · Compare · PDF Export</div>', unsafe_allow_html=True)
st.divider()

if not data_loaded: st.stop()

# ── TABS ───────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "Search", "Semantic vs TF-IDF", "Similarity Graph",
    "Citation Network", "Compare Papers", "Bookmarks", "History"
])

# ── TAB 1: SEARCH ─────────────────────────────────────────────
with tab1:
    c1,c2,c3 = st.columns([5,1,1])
    with c1: query = st.text_input("Query", placeholder="e.g. attention mechanism transformer NLP", label_visibility="collapsed")
    with c2: top_k = st.selectbox("N", [5,10,15,20], index=1, label_visibility="collapsed")
    with c3: use_mmr = st.checkbox("MMR", value=True)

    auto_sum = st.checkbox("Auto-summarize all results (slower)", value=False)

    if query:
        with st.spinner("Searching..."):
            results = semantic_search(query, top_k=top_k, mmr=use_mmr)
            add_to_history(query, len(results))

        c1,c2,c3 = st.columns([3,2,2])
        c1.success(f"{len(results)} papers found")
        if use_mmr: c2.caption("MMR diversity applied")

        if st.button("Export All as PDF"):
            with st.spinner("Generating PDF..."):
                sums = [summarize_bart(r['abstract']) for r in results]
                kws  = [get_keywords(r['abstract']) for r in results]
                buf  = make_pdf(query, results, sums, kws)
            st.download_button("Download PDF", data=buf,
                               file_name=f"{query[:20].replace(' ','_')}.pdf",
                               mime="application/pdf")

        if st.session_state.groq_key:
            if st.button("Groq: Research Trend Insights"):
                with st.spinner("Asking Groq..."):
                    insight = groq_insights(query, results)
                st.info(insight)

        st.divider()

        for i,r in enumerate(results):
            with st.container():
                c1,c2 = st.columns([8,2])
                with c1: st.markdown(f"**{i+1}. {r['title']}**")
                with c2: st.markdown(f'<span class="score-badge">{r["score"]:.4f}</span>', unsafe_allow_html=True)

                t1,t2,t3,t4 = st.tabs(["Abstract","BART Summary","Groq Summary","Keywords"])

                with t1: st.write(r['abstract'])
                with t2:
                    if st.button("Summarize", key=f"bs_{i}") or auto_sum:
                        with st.spinner(): s = summarize_bart(r['abstract'])
                        st.info(s)
                with t3:
                    if st.button("Ask Groq", key=f"gs_{i}"):
                        with st.spinner(): s = groq_summarize(r['abstract'])
                        st.info(s)
                with t4:
                    if st.button("Extract", key=f"kw_{i}"):
                        with st.spinner():
                            kb_kws = get_keywords(r['abstract'])
                            tf_kws = get_tfidf_keywords(r['abstract'])
                        st.markdown("**KeyBERT** " + " ".join([f'<span class="keyword-chip">{k}</span>' for k in kb_kws]), unsafe_allow_html=True)
                        st.markdown("**TF-IDF** " + " ".join([f'<span class="keyword-chip">{k}</span>' for k in tf_kws]), unsafe_allow_html=True)

                if st.button("Bookmark", key=f"bm_{i}"):
                    st.success("Saved!") if add_bookmark(r) else st.warning("Already saved")
                st.divider()

# ── TAB 2: SEMANTIC VS TFIDF ───────────────────────────────────
with tab2:
    st.markdown("**Semantic Search** (MiniLM + FAISS) vs **TF-IDF** (sklearn) — side by side comparison.")
    c1,c2 = st.columns([4,1])
    with c1: q = st.text_input("Query", placeholder="e.g. transformer attention", key="cmp_q", label_visibility="collapsed")
    with c2: k = st.selectbox("N", [5,10], key="cmp_k", label_visibility="collapsed")

    if q and st.button("Compare", type="primary"):
        with st.spinner("Running both searches..."):
            sem = semantic_search(q, top_k=k, mmr=False)
            tfi = tfidf_search(q, top_k=k)
            common = set(r['title'] for r in sem) & set(r['title'] for r in tfi)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Semantic Search**")
            st.caption("Meaning-based · all-MiniLM-L6-v2 + FAISS")
            for i,r in enumerate(sem):
                st.markdown(f"{i+1}. {r['title'][:65]}  \n`{r['score']:.4f}`")
        with c2:
            st.markdown("**TF-IDF Search**")
            st.caption("Keyword-based · sklearn TfidfVectorizer")
            for i,r in enumerate(tfi):
                st.markdown(f"{i+1}. {r['title'][:65]}  \n`{r['score']:.4f}`")

        st.divider()
        st.metric("Common Results", f"{len(common)}/{k}")
        if common:
            for t in common: st.markdown(f"  `{t[:80]}`")

        dark = st.session_state.dark_mode
        bg   = '#0d1117' if dark else '#f6f8fa'
        tc   = '#e6edf3' if dark else '#24292f'
        fig,axes = plt.subplots(1,2,figsize=(16,5))
        for ax,res,title,color in [
            (axes[0],sem,'Semantic Search','#58a6ff'),
            (axes[1],tfi,'TF-IDF Search','#f85149')
        ]:
            fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
            labels = [r['title'][:35]+'...' for r in res]
            sc     = [r['score'] for r in res]
            bars   = ax.barh(range(len(labels)),sc,color=color,alpha=0.85)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=8,color=tc)
            ax.set_title(title,fontweight='bold',color=tc,fontsize=11)
            ax.set_xlabel('Score',color=tc); ax.tick_params(colors=tc)
            for spine in ax.spines.values(): spine.set_color('#30363d')
            for bar,s in zip(bars,sc):
                ax.text(bar.get_width()+0.001,bar.get_y()+bar.get_height()/2,
                        f'{s:.3f}',va='center',fontsize=8,color=tc)
        plt.suptitle(f'Semantic vs TF-IDF · "{q}"',fontsize=12,fontweight='bold',color=tc)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Semantic search understands meaning — TF-IDF matches exact keywords only.")

# ── TAB 3: SIMILARITY GRAPH ────────────────────────────────────
with tab3:
    st.markdown("**Paper Similarity Graph** — Nodes = papers · Edges = cosine similarity > 0.72")
    c1,c2 = st.columns([4,1])
    with c1: sg_q = st.text_input("Query", placeholder="e.g. graph neural networks", key="sg_q", label_visibility="collapsed")
    with c2: sg_k = st.slider("Papers", 5,15,8, label_visibility="collapsed")

    if sg_q and st.button("Build Graph", type="primary"):
        with st.spinner("Building..."):
            res = semantic_search(sg_q, top_k=sg_k, mmr=False)
            fig = sim_graph_fig(res)
        st.pyplot(fig)
        for i,r in enumerate(res):
            st.markdown(f"`{i+1}.` {r['title']} — `{r['score']:.4f}`")

# ── TAB 4: CITATION NETWORK ────────────────────────────────────
with tab4:
    st.markdown("**Citation Network** — Enter 2–4 topics to see how papers across topics connect.")
    num = st.slider("Topics", 2,4,3)
    topics = [st.text_input(f"Topic {i+1}", key=f"t{i}",
                             placeholder=f"{'transformer attention' if i==0 else 'GAN image synthesis' if i==1 else 'reinforcement learning'}")
              for i in range(num)]
    ppt = st.slider("Papers per topic", 3,8,4)

    if all(topics) and st.button("Build Network", type="primary"):
        with st.spinner("Building..."):
            all_res = [semantic_search(t, top_k=ppt, mmr=True) for t in topics]
            fig     = citation_fig(topics, all_res)
        st.pyplot(fig)
        st.caption("Solid = same-topic · Dashed = cross-topic semantic overlap")

# ── TAB 5: COMPARE PAPERS ──────────────────────────────────────
with tab5:
    st.markdown("**Compare Two Papers** — Best match for each query, compared side by side.")
    c1,c2 = st.columns(2)
    with c1: q1 = st.text_input("Query A", placeholder="e.g. BERT language model", label_visibility="collapsed")
    with c2: q2 = st.text_input("Query B", placeholder="e.g. GPT generative model", label_visibility="collapsed")

    if q1 and q2 and st.button("Compare", type="primary", key="cmp_btn"):
        with st.spinner("Finding and comparing..."):
            r1 = semantic_search(q1, top_k=1, mmr=False)[0]
            r2 = semantic_search(q2, top_k=1, mmr=False)[0]
            cross_sim = cosine_similarity(embeddings[r1['idx']].reshape(1,-1),
                                          embeddings[r2['idx']].reshape(1,-1))[0][0]
            tfidf_sim = cos_sim(tfidf_vec.transform([r1['abstract']]),
                                tfidf_vec.transform([r2['abstract']]))[0][0]
            sum1  = summarize_bart(r1['abstract'])
            sum2  = summarize_bart(r2['abstract'])
            kw1   = get_keywords(r1['abstract'])
            kw2   = get_keywords(r2['abstract'])
            common = set(kw1) & set(kw2)

        c1,c2,c3 = st.columns(3)
        c1.metric("Semantic Similarity", f"{cross_sim:.4f}")
        c2.metric("TF-IDF Similarity",   f"{tfidf_sim:.4f}")
        c3.metric("Difference",           f"{abs(cross_sim-tfidf_sim):.4f}")
        if common: st.caption(f"Common keywords: {', '.join(common)}")

        if st.session_state.groq_key:
            if st.button("Groq AI Analysis"):
                with st.spinner("Asking Groq..."):
                    analysis = groq_compare(r1, r2, cross_sim)
                st.info(analysis)

        st.divider()
        c1,c2 = st.columns(2)
        for col,r,sum_,kws in [(c1,r1,sum1,kw1),(c2,r2,sum2,kw2)]:
            with col:
                st.markdown(f"**{r['title']}**")
                st.caption(f"Score: {r['score']:.4f}")
                st.info(sum_)
                st.markdown(" ".join([f'<span class="keyword-chip">{k}</span>' for k in kws]), unsafe_allow_html=True)
                st.markdown("")
                if st.button("Bookmark", key=f"bm_c_{r['idx']}"):
                    st.success("Saved!") if add_bookmark(r) else st.warning("Already saved")

# ── TAB 6: BOOKMARKS ───────────────────────────────────────────
with tab6:
    if not st.session_state.bookmarks:
        st.info("No bookmarks yet. Search papers and click Bookmark.")
    else:
        c1,c2 = st.columns([1,4])
        with c1: st.metric("Saved", len(st.session_state.bookmarks))
        with c2:
            if st.button("Clear All"):
                st.session_state.bookmarks=[]; st.rerun()

        if st.button("Export Bookmarks as PDF"):
            res = [{'title':b['title'],'abstract':b['abstract'],'score':b['score'],'idx':0}
                   for b in st.session_state.bookmarks]
            buf = make_pdf("My Bookmarks", res, [], [])
            st.download_button("Download PDF", data=buf, file_name="bookmarks.pdf",
                               mime="application/pdf")
        st.divider()
        for i,bm in enumerate(st.session_state.bookmarks):
            with st.expander(bm['title'][:80]):
                st.caption(f"Saved: {bm['saved_at']} · Score: {bm['score']:.4f}")
                st.write(bm['abstract'])
                if st.button("Remove", key=f"rm_{i}"):
                    st.session_state.bookmarks.pop(i); st.rerun()

# ── TAB 7: HISTORY ─────────────────────────────────────────────
with tab7:
    if not st.session_state.history:
        st.info("No search history yet.")
    else:
        c1,c2 = st.columns([1,4])
        with c1: st.metric("Searches", len(st.session_state.history))
        with c2:
            if st.button("Clear History"):
                st.session_state.history=[]; st.rerun()
        st.divider()
        for h in st.session_state.history:
            c1,c2,c3 = st.columns([5,2,2])
            c1.markdown(f"`{h['query']}`")
            c2.caption(f"{h['results']} results")
            c3.caption(h['timestamp'])