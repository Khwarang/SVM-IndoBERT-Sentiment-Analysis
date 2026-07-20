# import re
# from collections import Counter
# from datetime import datetime

# import streamlit as st
# import altair as alt
# import joblib
# import torch
# import pandas as pd
# import numpy as np
# import yt_dlp
# from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.feature_extraction.text import TfidfVectorizer
# from youtube_comment_downloader import YoutubeCommentDownloader

# try:
#     from transformers import AutoTokenizer, AutoModel
# except ImportError:
#     from transformers import AutoTokenizer as _AutoTokenizer
#     from transformers import AutoModel as _AutoModel
#     AutoTokenizer = _AutoTokenizer
#     AutoModel = _AutoModel

# # ============================================================
# # Konfigurasi halaman
# # ============================================================
# st.set_page_config(
#     page_title="Analisis Sentimen Komentar YouTube",
#     page_icon="💬",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ============================================================
# # Palet warna sentimen (dipakai untuk badge & chart)
# # ============================================================
# CHART_COLORS = {
#     "positif": "#10B981",
#     "negatif": "#F43F5E",
#     "netral": "#F59E0B",
# }
# BADGE_STYLES = {
#     "positif": ("#10B981", "#FFFFFF"),
#     "negatif": ("#F43F5E", "#FFFFFF"),
#     "netral": ("#FDE68A", "#92400E"),
# }
# SENTIMENT_ALIASES = {
#     "positif": "positif", "positive": "positif", "pos": "positif",
#     "negatif": "negatif", "negative": "negatif", "neg": "negatif",
#     "netral": "netral", "neutral": "netral", "netral/netral": "netral",
# }
# DEFAULT_CHART_COLOR = "#94A3B8"
# DEFAULT_BADGE_STYLE = ("#E2E8F0", "#334155")


# def normalize_sentiment(label: str) -> str:
#     return SENTIMENT_ALIASES.get(str(label).strip().lower(), str(label).strip().lower())


# def badge_html(label: str) -> str:
#     bg, fg = BADGE_STYLES.get(normalize_sentiment(label), DEFAULT_BADGE_STYLE)
#     return f'<span class="badge" style="background:{bg};color:{fg};">{label}</span>'


# # ============================================================
# # Styling global
# # ============================================================
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

# html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
# h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; }

# :root {
#   --brand: #0F766E;
#   --ink: #0F172A;
#   --muted: #64748B;
#   --border: #E2E8F0;
#   --bg-soft: #F8FAFC;
# }

# .block-container { padding-top: 1.6rem; max-width: 1180px; }

# .app-header { display:flex; align-items:center; gap:.7rem; margin-bottom:.2rem; }
# .app-header .icon { font-size: 2rem; line-height:1; }
# .app-title { font-family:'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--ink); margin:0; }
# .app-subtitle { color: var(--muted); font-size: .92rem; margin-top:.1rem; }

# .spectrum-bar {
#   height: 6px; border-radius: 999px; margin: .9rem 0 1.5rem 0;
#   background: linear-gradient(90deg, #F43F5E 0%, #F59E0B 50%, #10B981 100%);
# }

# .section-eyebrow {
#   text-transform: uppercase; letter-spacing: .08em; font-size: .72rem;
#   font-weight: 700; color: var(--brand); margin: 0 0 .2rem 0;
# }

# .stat-chip {
#   display:inline-flex; align-items:center; gap:.3rem; font-size:.78rem; color:var(--muted);
#   background: var(--bg-soft); border:1px solid var(--border); border-radius: 999px;
#   padding: .2rem .6rem; margin: .12rem .3rem .12rem 0;
# }

# .badge {
#   display:inline-block; font-size:.76rem; font-weight:700; padding:.22rem .7rem;
#   border-radius:999px;
# }

# .empty-state {
#   border: 1.5px dashed var(--border); border-radius: 14px; padding: 2.4rem 1.5rem;
#   text-align:center; color: var(--muted); background: var(--bg-soft);
# }
# .empty-state .icon { font-size: 2.2rem; margin-bottom:.5rem; display:block; }
# .empty-state b { color: var(--ink); }

# div.stButton > button { border-radius: 8px; font-weight: 600; }

# [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }

# [data-testid="stVerticalBlockBorderWrapper"] {
#   transition: box-shadow .15s ease;
# }
# [data-testid="stVerticalBlockBorderWrapper"]:hover {
#   box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
# }
# </style>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class="app-header">
#   <div class="icon">💬</div>
#   <div>
#     <p class="app-title">Analisis Sentimen Komentar YouTube</p>
#     <p class="app-subtitle">Prototipe IndoBERT + SVM — cari atau tempel link video, tarik komentarnya, lalu lihat sebaran sentimen dan topik pembahasannya.</p>
#   </div>
# </div>
# <div class="spectrum-bar"></div>
# """, unsafe_allow_html=True)


# # ============================================================
# # Load model & komponen (cache resource)
# # ============================================================
# @st.cache_resource(show_spinner="Memuat model IndoBERT + SVM...")
# def load_model():
#     svm = joblib.load('svm_model_final.pkl')
#     le = joblib.load('label_encoder.pkl')
#     tokenizer = AutoTokenizer.from_pretrained('model_indobert_deploy')
#     bert = AutoModel.from_pretrained('model_indobert_deploy')
#     return svm, le, tokenizer, bert

# svm, le, tokenizer, bert = load_model()


# @st.cache_resource
# def get_comment_downloader():
#     return YoutubeCommentDownloader()


# def fetch_youtube_comments(video_url: str, max_comments: int = 50):
#     downloader = get_comment_downloader()
#     comments = []
#     for item in downloader.get_comments_from_url(video_url, sort_by=1, language='id', sleep=0.1):
#         text = item.get('text', '') if isinstance(item, dict) else ''
#         if text:
#             comments.append(text)
#             if len(comments) >= max_comments:
#                 break
#     return comments


# def parse_upload_date(value):
#     if not value:
#         return None
#     if isinstance(value, (int, float)):
#         try:
#             return datetime.fromtimestamp(int(value))
#         except Exception:
#             return None
#     if isinstance(value, str):
#         if len(value) == 8:
#             try:
#                 return datetime.strptime(value, '%Y%m%d')
#             except ValueError:
#                 return None
#         try:
#             return datetime.fromisoformat(value)
#         except ValueError:
#             return None
#     return None


# def search_youtube_videos(query: str, max_results: int = 10, sort_mode: str = 'relevance'):
#     if not query.strip():
#         return []

#     ydl_opts = {
#         'quiet': True,
#         'skip_download': True,
#         'extract_flat': False,
#         'noplaylist': True,
#         'no_warnings': True,
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(f'ytsearch{max_results}:{query}', download=False)

#     entries = info.get('entries', []) if isinstance(info, dict) else []
#     results = []
#     for entry in entries:
#         if not entry:
#             continue

#         title = entry.get('title') or 'Tanpa judul'
#         video_id = entry.get('id') or ''
#         url = entry.get('webpage_url') or ''
#         if not url and video_id:
#             url = f'https://www.youtube.com/watch?v={video_id}'

#         if not url:
#             continue

#         thumbnail = entry.get('thumbnail') or ''
#         view_count = entry.get('view_count') or 0
#         comment_count = entry.get('comment_count') or 0
#         upload_date = entry.get('upload_date') or ''
#         release_timestamp = entry.get('release_timestamp')
#         upload_dt = parse_upload_date(upload_date) or parse_upload_date(release_timestamp)

#         results.append({
#             'title': title,
#             'url': url,
#             'thumbnail': thumbnail,
#             'view_count': int(view_count) if isinstance(view_count, (int, float)) else 0,
#             'comment_count': int(comment_count) if isinstance(comment_count, (int, float)) else 0,
#             'upload_date': upload_date,
#             'upload_dt': upload_dt,
#         })

#     if sort_mode == 'views':
#         results.sort(key=lambda x: x['view_count'], reverse=True)
#     elif sort_mode == 'newest':
#         results.sort(key=lambda x: x['upload_dt'] or datetime.min, reverse=True)
#     elif sort_mode == 'comments':
#         results.sort(key=lambda x: x['comment_count'], reverse=True)

#     return results


# def get_video_info(video_url: str):
#     if not video_url.strip():
#         return None

#     ydl_opts = {
#         'quiet': True,
#         'skip_download': True,
#         'extract_flat': False,
#         'noplaylist': True,
#         'no_warnings': True,
#     }

#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(video_url, download=False)

#     if not isinstance(info, dict):
#         return None

#     video_id = info.get('id') or ''
#     title = info.get('title') or 'Tanpa judul'
#     thumbnail = info.get('thumbnail') or ''
#     description = info.get('description') or ''
#     uploader = info.get('uploader') or ''
#     view_count = info.get('view_count') or 0
#     upload_date = info.get('upload_date') or ''
#     webpage_url = info.get('webpage_url') or video_url

#     return {
#         'title': title,
#         'url': webpage_url,
#         'thumbnail': thumbnail,
#         'description': description,
#         'uploader': uploader,
#         'view_count': int(view_count) if isinstance(view_count, (int, float)) else 0,
#         'upload_date': upload_date,
#         'video_id': video_id,
#     }


# def extract_topics(comments: list[str], top_n: int = 8):
#     stopwords = {
#         'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'sangat', 'untuk', 'pada', 'juga',
#         'akan', 'adalah', 'saya', 'aku', 'kamu', 'kami', 'mereka', 'tidak', 'bisa', 'sudah',
#         'apa', 'siapa', 'berapa', 'bagus', 'buruk', 'baik', 'jelek', 'video', 'link', 'youtube',
#         'comment', 'komentar', 'saya', 'sangat', 'sekali', 'lebih', 'satu', 'dua', 'tiga', 'empat',
#         'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh', 'kalo', 'kalau', 'ya', 'iya',
#         'terus', 'lagi', 'saja', 'semua', 'setelah', 'sebelum', 'karena', 'jadi', 'sering', 'suka',
#         'mau', 'harus', 'boleh', 'nanti', 'dengan', 'sini', 'situ', 'sana', 'mungkin', 'lebih', 'paling'
#     }

#     cleaned_comments = []
#     for comment in comments:
#         if not comment or not str(comment).strip():
#             continue
#         cleaned = re.sub(r'[^a-z0-9\s]', ' ', str(comment).lower())
#         words = [w for w in cleaned.split() if len(w) > 2 and w not in stopwords]
#         cleaned_comments.append(' '.join(words))

#     if len(cleaned_comments) < 3:
#         return {'summaries': [], 'assignments': []}

#     vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, max_features=500)
#     X = vectorizer.fit_transform(cleaned_comments)

#     n_topics = min(4, max(2, len(cleaned_comments) // 5 + 1))
#     lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=100)
#     lda.fit(X)

#     topic_assignments = lda.transform(X).argmax(axis=1)
#     feature_names = vectorizer.get_feature_names_out()
#     summaries = []
#     for topic_idx, topic in enumerate(lda.components_):
#         top_words = [feature_names[i] for i in topic.argsort()[:-top_n - 1:-1]]
#         summaries.append({
#             'topic': f'Topik {topic_idx + 1}',
#             'keywords': ', '.join(top_words),
#         })

#     return {'summaries': summaries, 'assignments': topic_assignments}


# # ============================================================
# # Pipeline analisis (dipakai baik oleh mode "search" maupun "link")
# # ============================================================
# def run_prediction_pipeline(video_items, max_comments, top_n_topics):
#     all_comments, source_links, source_titles = [], [], []

#     progress = st.progress(0.0, text="Menyiapkan pengambilan komentar...")
#     for i, item in enumerate(video_items):
#         progress.progress(i / len(video_items), text=f"Mengambil komentar dari: {item['title'][:60]}")
#         try:
#             comments = fetch_youtube_comments(item['url'], max_comments=int(max_comments))
#         except Exception as e:
#             st.error(f"Gagal mengambil komentar dari **{item['title']}**: {e}")
#             continue

#         if comments:
#             all_comments.extend(comments)
#             source_links.extend([item['url']] * len(comments))
#             source_titles.extend([item['title']] * len(comments))
#     progress.progress(1.0, text="Selesai mengambil komentar.")
#     progress.empty()

#     if not all_comments:
#         st.markdown("""
#         <div class="empty-state">
#           <span class="icon">🕳️</span>
#           <b>Tidak ada komentar ditemukan</b><br/>
#           Coba video lain, atau naikkan batas maksimal komentar per video.
#         </div>
#         """, unsafe_allow_html=True)
#         return

#     with st.spinner(f"Memprediksi sentimen untuk {len(all_comments):,} komentar..."):
#         batch_size = 16
#         all_embeddings = []
#         all_preds = []

#         for start in range(0, len(all_comments), batch_size):
#             batch_comments = all_comments[start:start + batch_size]
#             if not batch_comments:
#                 continue

#             inputs = tokenizer(batch_comments, return_tensors='pt', truncation=True, max_length=256, padding=True)
#             with torch.no_grad():
#                 outputs = bert(**inputs)
#                 batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#             all_embeddings.append(batch_emb)
#             all_preds.append(svm.predict(batch_emb))

#         emb = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 0))
#         preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
#         labels = le.inverse_transform(preds)

#     df = pd.DataFrame({
#         'source_title': source_titles,
#         'source_link': source_links,
#         'comment': all_comments,
#         'sentiment': labels,
#     })

#     m1, m2 = st.columns(2)
#     m1.metric("Video dianalisis", len(video_items))
#     m2.metric("Komentar dianalisis", f"{len(all_comments):,}")

#     tab_data, tab_sentiment, tab_topics = st.tabs(
#         ["📋 Data Komentar", "📊 Distribusi Sentimen", "🧠 Topik Pembahasan"]
#     )

#     with tab_data:
#         st.dataframe(
#             df,
#             use_container_width=True,
#             hide_index=True,
#             column_config={
#                 "source_title": st.column_config.TextColumn("Video"),
#                 "source_link": st.column_config.LinkColumn("Link"),
#                 "comment": st.column_config.TextColumn("Komentar", width="large"),
#                 "sentiment": st.column_config.TextColumn("Sentimen"),
#             },
#         )

#     with tab_sentiment:
#         counts = df['sentiment'].value_counts()
#         total = len(df)

#         badge_cols = st.columns(len(counts))
#         for col, (label, count) in zip(badge_cols, counts.items()):
#             with col:
#                 st.markdown(badge_html(label), unsafe_allow_html=True)
#                 st.metric(label="", value=f"{count:,}", delta=f"{count / total:.0%} dari total", delta_color="off")

#         chart_df = counts.rename_axis('sentiment').reset_index(name='jumlah')
#         domain = list(chart_df['sentiment'])
#         range_ = [CHART_COLORS.get(normalize_sentiment(s), DEFAULT_CHART_COLOR) for s in domain]

#         chart = alt.Chart(chart_df).mark_bar(cornerRadius=6, size=48).encode(
#             x=alt.X('sentiment:N', title=None, sort='-y'),
#             y=alt.Y('jumlah:Q', title='Jumlah komentar'),
#             color=alt.Color('sentiment:N', scale=alt.Scale(domain=domain, range=range_), legend=None),
#             tooltip=[alt.Tooltip('sentiment:N', title='Sentimen'), alt.Tooltip('jumlah:Q', title='Jumlah')],
#         ).properties(height=320)
#         st.altair_chart(chart, use_container_width=True)

#     with tab_topics:
#         topics_result = extract_topics(all_comments, top_n=int(top_n_topics))
#         if topics_result['summaries']:
#             for t in topics_result['summaries']:
#                 keywords = [kw for kw in t['keywords'].split(', ') if kw]
#                 chips = "".join(f'<span class="stat-chip">{kw}</span>' for kw in keywords)
#                 st.markdown(f"**{t['topic']}**<br/>{chips}", unsafe_allow_html=True)
#                 st.write("")

#             topic_counts = pd.Series(topics_result['assignments']).value_counts().sort_index()
#             if not topic_counts.empty:
#                 topic_labels = {idx: f'Topik {idx + 1}' for idx in topic_counts.index}
#                 topic_chart_df = (
#                     topic_counts.rename(index=topic_labels).rename_axis('topik').reset_index(name='jumlah')
#                 )
#                 topic_chart = alt.Chart(topic_chart_df).mark_bar(cornerRadius=6, size=48, color="#0F766E").encode(
#                     x=alt.X('topik:N', title=None),
#                     y=alt.Y('jumlah:Q', title='Jumlah komentar'),
#                     tooltip=[alt.Tooltip('topik:N', title='Topik'), alt.Tooltip('jumlah:Q', title='Jumlah')],
#                 ).properties(height=280)
#                 st.altair_chart(topic_chart, use_container_width=True)
#         else:
#             st.info("Tidak ada topik yang bisa diekstraksi dari komentar ini.")


# # ============================================================
# # Session state default
# # ============================================================
# if 'selected_video_urls' not in st.session_state:
#     st.session_state.selected_video_urls = []
# if 'last_search_query' not in st.session_state:
#     st.session_state.last_search_query = ''
# if 'direct_link_submitted' not in st.session_state:
#     st.session_state.direct_link_submitted = False
# if 'mode_selection' not in st.session_state:
#     st.session_state.mode_selection = 'search'
# if 'search_query_input' not in st.session_state:
#     st.session_state.search_query_input = ''
# if 'direct_video_url_input' not in st.session_state:
#     st.session_state.direct_video_url_input = ''
# if 'max_results_input' not in st.session_state:
#     st.session_state.max_results_input = 8
# if 'sort_mode_input' not in st.session_state:
#     st.session_state.sort_mode_input = 'relevance'
# if 'max_comments_input' not in st.session_state:
#     st.session_state.max_comments_input = 50
# if 'results' not in st.session_state:
#     st.session_state.results = []

# top_n_topics = 10

# # ============================================================
# # Sidebar — sumber video & pengaturan
# # ============================================================
# with st.sidebar:
#     st.markdown("### 🔎 Sumber Video")
#     st.caption("Pilih cara mengambil video yang ingin dianalisis.")

#     mode = st.radio(
#         "Cara mengambil video",
#         options=["search", "link"],
#         format_func=lambda x: "Cari topik" if x == "search" else "Tempel link",
#         horizontal=True,
#         key='mode_selection',
#         label_visibility="collapsed",
#     )

#     if mode == "search":
#         search_query = st.text_input(
#             "Kata kunci/topik video",
#             placeholder="Contoh: review hp samsung",
#             help="Masukkan topik yang ingin dicari di YouTube.",
#             key='search_query_input'
#         )
#         with st.expander("⚙️ Pengaturan pencarian"):
#             max_results = st.number_input(
#                 "Jumlah video yang ditampilkan",
#                 min_value=5,
#                 max_value=20,
#                 value=st.session_state.max_results_input,
#                 step=1,
#                 key='max_results_input'
#             )
#             sort_mode = st.selectbox(
#                 "Urutkan hasil",
#                 options=["relevance", "views", "newest", "comments"],
#                 format_func=lambda x: {
#                     'relevance': 'Paling relevan',
#                     'views': 'Paling banyak ditonton',
#                     'newest': 'Paling baru',
#                     'comments': 'Paling banyak komentar',
#                 }[x],
#                 key='sort_mode_input'
#             )
#     else:
#         direct_video_url = st.text_input(
#             "Link video YouTube",
#             placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
#             help="Masukkan satu link video YouTube yang ingin dianalisis komentarnya.",
#             key='direct_video_url_input'
#         )

#     max_comments = st.number_input(
#         "Maksimal komentar per video",
#         min_value=10,
#         max_value=200,
#         value=st.session_state.max_comments_input,
#         step=10,
#         key='max_comments_input'
#     )

#     st.write("")
#     if mode == "search":
#         search_clicked = st.button("🔍 Cari Video", type="primary", use_container_width=True)
#     else:
#         search_clicked = st.button("▶️ Analisis Link", type="primary", use_container_width=True, key="analyze_direct_link")

#     st.divider()
#     reset_clicked = st.button("↺ Reset semua", use_container_width=True)

# # --- Logika Reset ---
# if reset_clicked:
#     st.session_state.selected_video_urls = []
#     st.session_state.last_search_query = ''
#     st.session_state.results = []
#     st.session_state.direct_link_submitted = False
#     st.session_state.pop('direct_video_url', None)
#     st.session_state.pop('search_query', None)

#     for k in ['mode_selection', 'search_query_input', 'direct_video_url_input',
#               'max_results_input', 'sort_mode_input', 'max_comments_input']:
#         st.session_state.pop(k, None)

#     st.rerun()

# # --- Logika pencarian / analisis link ---
# if mode == "search":
#     if search_clicked:
#         if not search_query.strip():
#             st.sidebar.warning("Mohon masukkan kata kunci/topik terlebih dahulu.")
#             st.session_state.results = []
#         else:
#             with st.spinner("Mencari video yang relevan..."):
#                 st.session_state.results = search_youtube_videos(
#                     search_query, max_results=int(max_results), sort_mode=sort_mode
#                 )

#             if not st.session_state.results:
#                 st.info("Tidak ada video yang ditemukan untuk kata kunci tersebut.")
#             else:
#                 if search_query != st.session_state.get('last_search_query', ''):
#                     st.session_state.selected_video_urls = []
#                     st.session_state.last_search_query = search_query

# if mode == "link":
#     if search_clicked:
#         if not direct_video_url.strip():
#             st.sidebar.warning("Masukkan link video YouTube terlebih dahulu.")
#             st.session_state.results = []
#             st.session_state.direct_link_submitted = False
#         else:
#             with st.spinner("Mengambil detail video..."):
#                 info = get_video_info(direct_video_url.strip())
#             if info:
#                 st.session_state.results = [{
#                     'title': info['title'],
#                     'url': info['url'],
#                     'thumbnail': info['thumbnail'],
#                     'view_count': info['view_count'],
#                     'comment_count': 0,
#                     'upload_date': info['upload_date'],
#                     'upload_dt': None,
#                     'description': info['description'],
#                     'uploader': info['uploader'],
#                     'video_id': info['video_id'],
#                 }]
#                 st.session_state.direct_link_submitted = True
#             else:
#                 st.session_state.results = []
#                 st.session_state.direct_link_submitted = False
#                 st.warning("Tidak dapat mengambil detail video dari link yang dimasukkan.")
#     elif not direct_video_url.strip():
#         st.session_state.results = []
#         st.session_state.direct_link_submitted = False

# results = st.session_state.results

# # ============================================================
# # Area utama — hasil pencarian / detail video
# # ============================================================
# if not results:
#     st.markdown("""
#     <div class="empty-state">
#       <span class="icon">🎬</span>
#       <b>Belum ada video</b><br/>
#       Cari topik atau tempel link video YouTube di panel kiri untuk memulai.
#     </div>
#     """, unsafe_allow_html=True)
# else:
#     analyze_clicked = False
#     video_items_to_analyze = []

#     if mode == "link":
#         item = results[0]
#         st.markdown('<p class="section-eyebrow">Detail video</p>', unsafe_allow_html=True)
#         with st.container(border=True):
#             st.markdown(f"**{item['title']}**")
#             if item.get('video_id'):
#                 st.video(f"https://www.youtube.com/watch?v={item['video_id']}")
#             st.markdown(
#                 f'<span class="stat-chip">👁️ {item["view_count"]:,} tontonan</span>'
#                 f'<span class="stat-chip">📅 {item["upload_date"] or "-"}</span>'
#                 + (f'<span class="stat-chip">👤 {item["uploader"]}</span>' if item.get('uploader') else ''),
#                 unsafe_allow_html=True,
#             )
#             if item.get('description'):
#                 with st.expander("Deskripsi"):
#                     st.write(item['description'])

#         st.write("")
#         analyze_clicked = st.button("🚀 Ambil & Prediksi Sentimen", type="primary",
#                                      use_container_width=True, key="process_direct_video")
#         video_items_to_analyze = [item]

#     else:
#         st.markdown('<p class="section-eyebrow">Hasil pencarian</p>', unsafe_allow_html=True)
#         st.subheader(f"{len(results)} video ditemukan")

#         cols_per_row = 3
#         for row_start in range(0, len(results), cols_per_row):
#             row_items = results[row_start: row_start + cols_per_row]
#             cols = st.columns(cols_per_row)
#             for col_i, item in enumerate(row_items):
#                 idx = row_start + col_i
#                 with cols[col_i]:
#                     with st.container(border=True, key=f"video_card_{idx}"):
#                         if item.get('thumbnail'):
#                             st.image(item['thumbnail'], use_container_width=True)

#                         title = item['title']
#                         st.markdown(f"**{title[:70]}{'…' if len(title) > 70 else ''}**")
#                         if item.get('uploader'):
#                             st.caption(f"👤 {item['uploader']}")

#                         st.markdown(
#                             f'<span class="stat-chip">👁️ {item["view_count"]:,}</span>'
#                             f'<span class="stat-chip">💬 {item["comment_count"]:,}</span>'
#                             f'<span class="stat-chip">📅 {item["upload_date"] or "-"}</span>',
#                             unsafe_allow_html=True,
#                         )

#                         if item.get('description'):
#                             with st.expander("Deskripsi"):
#                                 st.write(item['description'])

#                         is_selected = item['url'] in st.session_state.selected_video_urls
#                         btn_label = "✅ Terpilih — batalkan" if is_selected else "➕ Pilih video"
#                         if st.button(
#                             btn_label,
#                             key=f"toggle_{idx}_{item['url']}",
#                             use_container_width=True,
#                             type="primary" if is_selected else "secondary",
#                         ):
#                             if item['url'] in st.session_state.selected_video_urls:
#                                 st.session_state.selected_video_urls.remove(item['url'])
#                             elif len(st.session_state.selected_video_urls) < 3:
#                                 st.session_state.selected_video_urls.append(item['url'])
#                             else:
#                                 st.warning("Maksimal 3 video saja yang bisa dipilih.")
#                             st.rerun()

#         # Highlight kartu yang sudah dipilih
#         selected_css = "".join(
#             f'.st-key-video_card_{idx} {{ border: 1.5px solid #10B981 !important; }}\n'
#             for idx, item in enumerate(results)
#             if item['url'] in st.session_state.selected_video_urls
#         )
#         if selected_css:
#             st.markdown(f"<style>{selected_css}</style>", unsafe_allow_html=True)

#         st.divider()
#         n_selected = len(st.session_state.selected_video_urls)
#         if n_selected:
#             st.markdown(f"**{n_selected}/3** video dipilih")
#             st.progress(n_selected / 3)
#             selected_results = [item for item in results if item['url'] in st.session_state.selected_video_urls]
#             analyze_clicked = st.button(
#                 "🚀 Ambil & Prediksi Sentimen", type="primary",
#                 use_container_width=True, key="process_selected_videos",
#             )
#             video_items_to_analyze = selected_results
#         else:
#             st.info("Pilih minimal 1 video (maksimal 3) untuk mulai analisis.")

#     if analyze_clicked and video_items_to_analyze:
#         st.divider()
#         st.markdown('<p class="section-eyebrow">Hasil analisis</p>', unsafe_allow_html=True)
#         run_prediction_pipeline(video_items_to_analyze, max_comments, top_n_topics)


# *********************************DARI GDRIVE*********************************


import re
from collections import Counter
from datetime import datetime

import streamlit as st
import altair as alt
import joblib
import torch
import pandas as pd
import numpy as np
import yt_dlp
import os
import gdown
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from youtube_comment_downloader import YoutubeCommentDownloader

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    from transformers import AutoTokenizer as _AutoTokenizer
    from transformers import AutoModel as _AutoModel
    AutoTokenizer = _AutoTokenizer
    AutoModel = _AutoModel

# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen Komentar YouTube",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Palet warna sentimen (dipakai untuk badge & chart)
# ============================================================
CHART_COLORS = {
    "positif": "#10B981",
    "negatif": "#F43F5E",
    "netral": "#F59E0B",
}
BADGE_STYLES = {
    "positif": ("#10B981", "#FFFFFF"),
    "negatif": ("#F43F5E", "#FFFFFF"),
    "netral": ("#FDE68A", "#92400E"),
}
SENTIMENT_ALIASES = {
    "positif": "positif", "positive": "positif", "pos": "positif",
    "negatif": "negatif", "negative": "negatif", "neg": "negatif",
    "netral": "netral", "neutral": "netral", "netral/netral": "netral",
}
DEFAULT_CHART_COLOR = "#94A3B8"
DEFAULT_BADGE_STYLE = ("#E2E8F0", "#334155")


def normalize_sentiment(label: str) -> str:
    return SENTIMENT_ALIASES.get(str(label).strip().lower(), str(label).strip().lower())


def badge_html(label: str) -> str:
    bg, fg = BADGE_STYLES.get(normalize_sentiment(label), DEFAULT_BADGE_STYLE)
    return f'<span class="badge" style="background:{bg};color:{fg};">{label}</span>'


# ============================================================
# Styling global
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; }

:root {
  --brand: #0F766E;
  --ink: #0F172A;
  --muted: #64748B;
  --border: #E2E8F0;
  --bg-soft: #F8FAFC;
}

.block-container { padding-top: 1.6rem; max-width: 1180px; }

.app-header { display:flex; align-items:center; gap:.7rem; margin-bottom:.2rem; }
.app-header .icon { font-size: 2rem; line-height:1; }
.app-title { font-family:'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--ink); margin:0; }
.app-subtitle { color: var(--muted); font-size: .92rem; margin-top:.1rem; }

.spectrum-bar {
  height: 6px; border-radius: 999px; margin: .9rem 0 1.5rem 0;
  background: linear-gradient(90deg, #F43F5E 0%, #F59E0B 50%, #10B981 100%);
}

.section-eyebrow {
  text-transform: uppercase; letter-spacing: .08em; font-size: .72rem;
  font-weight: 700; color: var(--brand); margin: 0 0 .2rem 0;
}

.stat-chip {
  display:inline-flex; align-items:center; gap:.3rem; font-size:.78rem; color:var(--muted);
  background: var(--bg-soft); border:1px solid var(--border); border-radius: 999px;
  padding: .2rem .6rem; margin: .12rem .3rem .12rem 0;
}

.badge {
  display:inline-block; font-size:.76rem; font-weight:700; padding:.22rem .7rem;
  border-radius:999px;
}

.empty-state {
  border: 1.5px dashed var(--border); border-radius: 14px; padding: 2.4rem 1.5rem;
  text-align:center; color: var(--muted); background: var(--bg-soft);
}
.empty-state .icon { font-size: 2.2rem; margin-bottom:.5rem; display:block; }
.empty-state b { color: var(--ink); }

div.stButton > button { border-radius: 8px; font-weight: 600; }

[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }

[data-testid="stVerticalBlockBorderWrapper"] {
  transition: box-shadow .15s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div class="icon">💬</div>
  <div>
    <p class="app-title">Analisis Sentimen Komentar YouTube</p>
    <p class="app-subtitle">Prototipe IndoBERT + SVM — cari atau tempel link video, tarik komentarnya, lalu lihat sebaran sentimen dan topik pembahasannya.</p>
  </div>
</div>
<div class="spectrum-bar"></div>
""", unsafe_allow_html=True)


# ============================================================
# Load model & komponen (cache resource)
# ============================================================
@st.cache_resource(show_spinner="Memuat model IndoBERT + SVM... (Jika baru pertama kali, ini memakan waktu beberapa menit untuk unduh model)")
def load_model():
    # --- MULAI KODE GDOWN ---
    model_dir = "model_indobert_deploy"
    model_path = f"{model_dir}/model.safetensors"
    
    # Buat foldernya jika belum ada
    os.makedirs(model_dir, exist_ok=True)
    
    # Cek apakah file sudah ada, jika belum, download dari GDrive
    if not os.path.exists(model_path):
        file_id = '1fASdjIaoPe-bIEq9RHZTvwmCpEo3uwhF'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    # --- SELESAI KODE GDOWN ---

    # Lanjutkan memuat model seperti biasa
    svm = joblib.load('svm_model_final.pkl')
    le = joblib.load('label_encoder.pkl')
    tokenizer = AutoTokenizer.from_pretrained('model_indobert_deploy')
    bert = AutoModel.from_pretrained('model_indobert_deploy')
    
    return svm, le, tokenizer, bert

svm, le, tokenizer, bert = load_model()


@st.cache_resource
def get_comment_downloader():
    return YoutubeCommentDownloader()


def fetch_youtube_comments(video_url: str, max_comments: int = 50):
    downloader = get_comment_downloader()
    comments = []
    for item in downloader.get_comments_from_url(video_url, sort_by=1, language='id', sleep=0.1):
        text = item.get('text', '') if isinstance(item, dict) else ''
        if text:
            comments.append(text)
            if len(comments) >= max_comments:
                break
    return comments


def parse_upload_date(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(int(value))
        except Exception:
            return None
    if isinstance(value, str):
        if len(value) == 8:
            try:
                return datetime.strptime(value, '%Y%m%d')
            except ValueError:
                return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def search_youtube_videos(query: str, max_results: int = 10, sort_mode: str = 'relevance'):
    if not query.strip():
        return []

    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': False,
        'noplaylist': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f'ytsearch{max_results}:{query}', download=False)

    entries = info.get('entries', []) if isinstance(info, dict) else []
    results = []
    for entry in entries:
        if not entry:
            continue

        title = entry.get('title') or 'Tanpa judul'
        video_id = entry.get('id') or ''
        url = entry.get('webpage_url') or ''
        if not url and video_id:
            url = f'https://www.youtube.com/watch?v={video_id}'

        if not url:
            continue

        thumbnail = entry.get('thumbnail') or ''
        view_count = entry.get('view_count') or 0
        comment_count = entry.get('comment_count') or 0
        upload_date = entry.get('upload_date') or ''
        release_timestamp = entry.get('release_timestamp')
        upload_dt = parse_upload_date(upload_date) or parse_upload_date(release_timestamp)

        results.append({
            'title': title,
            'url': url,
            'thumbnail': thumbnail,
            'view_count': int(view_count) if isinstance(view_count, (int, float)) else 0,
            'comment_count': int(comment_count) if isinstance(comment_count, (int, float)) else 0,
            'upload_date': upload_date,
            'upload_dt': upload_dt,
        })

    if sort_mode == 'views':
        results.sort(key=lambda x: x['view_count'], reverse=True)
    elif sort_mode == 'newest':
        results.sort(key=lambda x: x['upload_dt'] or datetime.min, reverse=True)
    elif sort_mode == 'comments':
        results.sort(key=lambda x: x['comment_count'], reverse=True)

    return results


def get_video_info(video_url: str):
    if not video_url.strip():
        return None

    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': False,
        'noplaylist': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    if not isinstance(info, dict):
        return None

    video_id = info.get('id') or ''
    title = info.get('title') or 'Tanpa judul'
    thumbnail = info.get('thumbnail') or ''
    description = info.get('description') or ''
    uploader = info.get('uploader') or ''
    view_count = info.get('view_count') or 0
    upload_date = info.get('upload_date') or ''
    webpage_url = info.get('webpage_url') or video_url

    return {
        'title': title,
        'url': webpage_url,
        'thumbnail': thumbnail,
        'description': description,
        'uploader': uploader,
        'view_count': int(view_count) if isinstance(view_count, (int, float)) else 0,
        'upload_date': upload_date,
        'video_id': video_id,
    }


def extract_topics(comments: list[str], top_n: int = 8):
    stopwords = {
        'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'sangat', 'untuk', 'pada', 'juga',
        'akan', 'adalah', 'saya', 'aku', 'kamu', 'kami', 'mereka', 'tidak', 'bisa', 'sudah',
        'apa', 'siapa', 'berapa', 'bagus', 'buruk', 'baik', 'jelek', 'video', 'link', 'youtube',
        'comment', 'komentar', 'saya', 'sangat', 'sekali', 'lebih', 'satu', 'dua', 'tiga', 'empat',
        'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh', 'kalo', 'kalau', 'ya', 'iya',
        'terus', 'lagi', 'saja', 'semua', 'setelah', 'sebelum', 'karena', 'jadi', 'sering', 'suka',
        'mau', 'harus', 'boleh', 'nanti', 'dengan', 'sini', 'situ', 'sana', 'mungkin', 'lebih', 'paling'
    }

    cleaned_comments = []
    for comment in comments:
        if not comment or not str(comment).strip():
            continue
        cleaned = re.sub(r'[^a-z0-9\s]', ' ', str(comment).lower())
        words = [w for w in cleaned.split() if len(w) > 2 and w not in stopwords]
        cleaned_comments.append(' '.join(words))

    if len(cleaned_comments) < 3:
        return {'summaries': [], 'assignments': []}

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, max_features=500)
    X = vectorizer.fit_transform(cleaned_comments)

    n_topics = min(4, max(2, len(cleaned_comments) // 5 + 1))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=100)
    lda.fit(X)

    topic_assignments = lda.transform(X).argmax(axis=1)
    feature_names = vectorizer.get_feature_names_out()
    summaries = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-top_n - 1:-1]]
        summaries.append({
            'topic': f'Topik {topic_idx + 1}',
            'keywords': ', '.join(top_words),
        })

    return {'summaries': summaries, 'assignments': topic_assignments}


# ============================================================
# Pipeline analisis (dipakai baik oleh mode "search" maupun "link")
# ============================================================
def run_prediction_pipeline(video_items, max_comments, top_n_topics):
    all_comments, source_links, source_titles = [], [], []

    progress = st.progress(0.0, text="Menyiapkan pengambilan komentar...")
    for i, item in enumerate(video_items):
        progress.progress(i / len(video_items), text=f"Mengambil komentar dari: {item['title'][:60]}")
        try:
            comments = fetch_youtube_comments(item['url'], max_comments=int(max_comments))
        except Exception as e:
            st.error(f"Gagal mengambil komentar dari **{item['title']}**: {e}")
            continue

        if comments:
            all_comments.extend(comments)
            source_links.extend([item['url']] * len(comments))
            source_titles.extend([item['title']] * len(comments))
    progress.progress(1.0, text="Selesai mengambil komentar.")
    progress.empty()

    if not all_comments:
        st.markdown("""
        <div class="empty-state">
          <span class="icon">🕳️</span>
          <b>Tidak ada komentar ditemukan</b><br/>
          Coba video lain, atau naikkan batas maksimal komentar per video.
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner(f"Memprediksi sentimen untuk {len(all_comments):,} komentar..."):
        batch_size = 16
        all_embeddings = []
        all_preds = []

        for start in range(0, len(all_comments), batch_size):
            batch_comments = all_comments[start:start + batch_size]
            if not batch_comments:
                continue

            inputs = tokenizer(batch_comments, return_tensors='pt', truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                outputs = bert(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_emb)
            all_preds.append(svm.predict(batch_emb))

        emb = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 0))
        preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        labels = le.inverse_transform(preds)

    df = pd.DataFrame({
        'source_title': source_titles,
        'source_link': source_links,
        'comment': all_comments,
        'sentiment': labels,
    })

    m1, m2 = st.columns(2)
    m1.metric("Video dianalisis", len(video_items))
    m2.metric("Komentar dianalisis", f"{len(all_comments):,}")

    tab_data, tab_sentiment, tab_topics = st.tabs(
        ["📋 Data Komentar", "📊 Distribusi Sentimen", "🧠 Topik Pembahasan"]
    )

    with tab_data:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "source_title": st.column_config.TextColumn("Video"),
                "source_link": st.column_config.LinkColumn("Link"),
                "comment": st.column_config.TextColumn("Komentar", width="large"),
                "sentiment": st.column_config.TextColumn("Sentimen"),
            },
        )

    with tab_sentiment:
        counts = df['sentiment'].value_counts()
        total = len(df)

        badge_cols = st.columns(len(counts))
        for col, (label, count) in zip(badge_cols, counts.items()):
            with col:
                st.markdown(badge_html(label), unsafe_allow_html=True)
                st.metric(label="", value=f"{count:,}", delta=f"{count / total:.0%} dari total", delta_color="off")

        chart_df = counts.rename_axis('sentiment').reset_index(name='jumlah')
        domain = list(chart_df['sentiment'])
        range_ = [CHART_COLORS.get(normalize_sentiment(s), DEFAULT_CHART_COLOR) for s in domain]

        chart = alt.Chart(chart_df).mark_bar(cornerRadius=6, size=48).encode(
            x=alt.X('sentiment:N', title=None, sort='-y'),
            y=alt.Y('jumlah:Q', title='Jumlah komentar'),
            color=alt.Color('sentiment:N', scale=alt.Scale(domain=domain, range=range_), legend=None),
            tooltip=[alt.Tooltip('sentiment:N', title='Sentimen'), alt.Tooltip('jumlah:Q', title='Jumlah')],
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)

    with tab_topics:
        topics_result = extract_topics(all_comments, top_n=int(top_n_topics))
        if topics_result['summaries']:
            for t in topics_result['summaries']:
                keywords = [kw for kw in t['keywords'].split(', ') if kw]
                chips = "".join(f'<span class="stat-chip">{kw}</span>' for kw in keywords)
                st.markdown(f"**{t['topic']}**<br/>{chips}", unsafe_allow_html=True)
                st.write("")

            topic_counts = pd.Series(topics_result['assignments']).value_counts().sort_index()
            if not topic_counts.empty:
                topic_labels = {idx: f'Topik {idx + 1}' for idx in topic_counts.index}
                topic_chart_df = (
                    topic_counts.rename(index=topic_labels).rename_axis('topik').reset_index(name='jumlah')
                )
                topic_chart = alt.Chart(topic_chart_df).mark_bar(cornerRadius=6, size=48, color="#0F766E").encode(
                    x=alt.X('topik:N', title=None),
                    y=alt.Y('jumlah:Q', title='Jumlah komentar'),
                    tooltip=[alt.Tooltip('topik:N', title='Topik'), alt.Tooltip('jumlah:Q', title='Jumlah')],
                ).properties(height=280)
                st.altair_chart(topic_chart, use_container_width=True)
        else:
            st.info("Tidak ada topik yang bisa diekstraksi dari komentar ini.")


# ============================================================
# Session state default
# ============================================================
if 'selected_video_urls' not in st.session_state:
    st.session_state.selected_video_urls = []
if 'last_search_query' not in st.session_state:
    st.session_state.last_search_query = ''
if 'direct_link_submitted' not in st.session_state:
    st.session_state.direct_link_submitted = False
if 'mode_selection' not in st.session_state:
    st.session_state.mode_selection = 'search'
if 'search_query_input' not in st.session_state:
    st.session_state.search_query_input = ''
if 'direct_video_url_input' not in st.session_state:
    st.session_state.direct_video_url_input = ''
if 'max_results_input' not in st.session_state:
    st.session_state.max_results_input = 8
if 'sort_mode_input' not in st.session_state:
    st.session_state.sort_mode_input = 'relevance'
if 'max_comments_input' not in st.session_state:
    st.session_state.max_comments_input = 50
if 'results' not in st.session_state:
    st.session_state.results = []

top_n_topics = 10

# ============================================================
# Sidebar — sumber video & pengaturan
# ============================================================
with st.sidebar:
    st.markdown("### 🔎 Sumber Video")
    st.caption("Pilih cara mengambil video yang ingin dianalisis.")

    mode = st.radio(
        "Cara mengambil video",
        options=["search", "link"],
        format_func=lambda x: "Cari topik" if x == "search" else "Tempel link",
        horizontal=True,
        key='mode_selection',
        label_visibility="collapsed",
    )

    if mode == "search":
        search_query = st.text_input(
            "Kata kunci/topik video",
            placeholder="Contoh: review hp samsung",
            help="Masukkan topik yang ingin dicari di YouTube.",
            key='search_query_input'
        )
        with st.expander("⚙️ Pengaturan pencarian"):
            max_results = st.number_input(
                "Jumlah video yang ditampilkan",
                min_value=5,
                max_value=20,
                value=st.session_state.max_results_input,
                step=1,
                key='max_results_input'
            )
            sort_mode = st.selectbox(
                "Urutkan hasil",
                options=["relevance", "views", "newest", "comments"],
                format_func=lambda x: {
                    'relevance': 'Paling relevan',
                    'views': 'Paling banyak ditonton',
                    'newest': 'Paling baru',
                    'comments': 'Paling banyak komentar',
                }[x],
                key='sort_mode_input'
            )
    else:
        direct_video_url = st.text_input(
            "Link video YouTube",
            placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
            help="Masukkan satu link video YouTube yang ingin dianalisis komentarnya.",
            key='direct_video_url_input'
        )

    max_comments = st.number_input(
        "Maksimal komentar per video",
        min_value=10,
        max_value=200,
        value=st.session_state.max_comments_input,
        step=10,
        key='max_comments_input'
    )

    st.write("")
    if mode == "search":
        search_clicked = st.button("🔍 Cari Video", type="primary", use_container_width=True)
    else:
        search_clicked = st.button("▶️ Analisis Link", type="primary", use_container_width=True, key="analyze_direct_link")

    st.divider()
    reset_clicked = st.button("↺ Reset semua", use_container_width=True)

# --- Logika Reset ---
if reset_clicked:
    st.session_state.selected_video_urls = []
    st.session_state.last_search_query = ''
    st.session_state.results = []
    st.session_state.direct_link_submitted = False
    st.session_state.pop('direct_video_url', None)
    st.session_state.pop('search_query', None)

    for k in ['mode_selection', 'search_query_input', 'direct_video_url_input',
              'max_results_input', 'sort_mode_input', 'max_comments_input']:
        st.session_state.pop(k, None)

    st.rerun()

# --- Logika pencarian / analisis link ---
if mode == "search":
    if search_clicked:
        if not search_query.strip():
            st.sidebar.warning("Mohon masukkan kata kunci/topik terlebih dahulu.")
            st.session_state.results = []
        else:
            with st.spinner("Mencari video yang relevan..."):
                st.session_state.results = search_youtube_videos(
                    search_query, max_results=int(max_results), sort_mode=sort_mode
                )

            if not st.session_state.results:
                st.info("Tidak ada video yang ditemukan untuk kata kunci tersebut.")
            else:
                if search_query != st.session_state.get('last_search_query', ''):
                    st.session_state.selected_video_urls = []
                    st.session_state.last_search_query = search_query

if mode == "link":
    if search_clicked:
        if not direct_video_url.strip():
            st.sidebar.warning("Masukkan link video YouTube terlebih dahulu.")
            st.session_state.results = []
            st.session_state.direct_link_submitted = False
        else:
            with st.spinner("Mengambil detail video..."):
                info = get_video_info(direct_video_url.strip())
            if info:
                st.session_state.results = [{
                    'title': info['title'],
                    'url': info['url'],
                    'thumbnail': info['thumbnail'],
                    'view_count': info['view_count'],
                    'comment_count': 0,
                    'upload_date': info['upload_date'],
                    'upload_dt': None,
                    'description': info['description'],
                    'uploader': info['uploader'],
                    'video_id': info['video_id'],
                }]
                st.session_state.direct_link_submitted = True
            else:
                st.session_state.results = []
                st.session_state.direct_link_submitted = False
                st.warning("Tidak dapat mengambil detail video dari link yang dimasukkan.")
    elif not direct_video_url.strip():
        st.session_state.results = []
        st.session_state.direct_link_submitted = False

results = st.session_state.results

# ============================================================
# Area utama — hasil pencarian / detail video
# ============================================================
if not results:
    st.markdown("""
    <div class="empty-state">
      <span class="icon">🎬</span>
      <b>Belum ada video</b><br/>
      Cari topik atau tempel link video YouTube di panel kiri untuk memulai.
    </div>
    """, unsafe_allow_html=True)
else:
    analyze_clicked = False
    video_items_to_analyze = []

    if mode == "link":
        item = results[0]
        st.markdown('<p class="section-eyebrow">Detail video</p>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            if item.get('video_id'):
                st.video(f"https://www.youtube.com/watch?v={item['video_id']}")
            st.markdown(
                f'<span class="stat-chip">👁️ {item["view_count"]:,} tontonan</span>'
                f'<span class="stat-chip">📅 {item["upload_date"] or "-"}</span>'
                + (f'<span class="stat-chip">👤 {item["uploader"]}</span>' if item.get('uploader') else ''),
                unsafe_allow_html=True,
            )
            if item.get('description'):
                with st.expander("Deskripsi"):
                    st.write(item['description'])

        st.write("")
        analyze_clicked = st.button("🚀 Ambil & Prediksi Sentimen", type="primary",
                                     use_container_width=True, key="process_direct_video")
        video_items_to_analyze = [item]

    else:
        st.markdown('<p class="section-eyebrow">Hasil pencarian</p>', unsafe_allow_html=True)
        st.subheader(f"{len(results)} video ditemukan")

        cols_per_row = 3
        for row_start in range(0, len(results), cols_per_row):
            row_items = results[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col_i, item in enumerate(row_items):
                idx = row_start + col_i
                with cols[col_i]:
                    with st.container(border=True, key=f"video_card_{idx}"):
                        if item.get('thumbnail'):
                            st.image(item['thumbnail'], use_container_width=True)

                        title = item['title']
                        st.markdown(f"**{title[:70]}{'…' if len(title) > 70 else ''}**")
                        if item.get('uploader'):
                            st.caption(f"👤 {item['uploader']}")

                        st.markdown(
                            f'<span class="stat-chip">👁️ {item["view_count"]:,}</span>'
                            f'<span class="stat-chip">💬 {item["comment_count"]:,}</span>'
                            f'<span class="stat-chip">📅 {item["upload_date"] or "-"}</span>',
                            unsafe_allow_html=True,
                        )

                        if item.get('description'):
                            with st.expander("Deskripsi"):
                                st.write(item['description'])

                        is_selected = item['url'] in st.session_state.selected_video_urls
                        btn_label = "✅ Terpilih — batalkan" if is_selected else "➕ Pilih video"
                        if st.button(
                            btn_label,
                            key=f"toggle_{idx}_{item['url']}",
                            use_container_width=True,
                            type="primary" if is_selected else "secondary",
                        ):
                            if item['url'] in st.session_state.selected_video_urls:
                                st.session_state.selected_video_urls.remove(item['url'])
                            elif len(st.session_state.selected_video_urls) < 3:
                                st.session_state.selected_video_urls.append(item['url'])
                            else:
                                st.warning("Maksimal 3 video saja yang bisa dipilih.")
                            st.rerun()

        # Highlight kartu yang sudah dipilih
        selected_css = "".join(
            f'.st-key-video_card_{idx} {{ border: 1.5px solid #10B981 !important; }}\n'
            for idx, item in enumerate(results)
            if item['url'] in st.session_state.selected_video_urls
        )
        if selected_css:
            st.markdown(f"<style>{selected_css}</style>", unsafe_allow_html=True)

        st.divider()
        n_selected = len(st.session_state.selected_video_urls)
        if n_selected:
            st.markdown(f"**{n_selected}/3** video dipilih")
            st.progress(n_selected / 3)
            selected_results = [item for item in results if item['url'] in st.session_state.selected_video_urls]
            analyze_clicked = st.button(
                "🚀 Ambil & Prediksi Sentimen", type="primary",
                use_container_width=True, key="process_selected_videos",
            )
            video_items_to_analyze = selected_results
        else:
            st.info("Pilih minimal 1 video (maksimal 3) untuk mulai analisis.")

    if analyze_clicked and video_items_to_analyze:
        st.divider()
        st.markdown('<p class="section-eyebrow">Hasil analisis</p>', unsafe_allow_html=True)
        run_prediction_pipeline(video_items_to_analyze, max_comments, top_n_topics)