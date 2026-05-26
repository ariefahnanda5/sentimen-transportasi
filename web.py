import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from streamlit_option_menu import option_menu  
from google_play_scraper import reviews, Sort
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor
import os

@st.cache_resource
def load_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab') 
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True) 
        nltk.download('stopwords', quiet=True)

load_nltk_resources()

st.set_page_config(page_title="Analisis Sentimen Transportasi Online", layout="wide")

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### 📊 Menu Utama")
    selected = option_menu(
        menu_title=None,
        options=[
            "Beranda & Panduan", 
            "Input & Scraping Data", 
            "Pembersihan Data", 
            "Support Vector Machine (SVM)",         
            "Naive Bayes", 
            "Dashboard & Visualisasi"
        ],
        icons=["house", "cloud-download", "brush", "cpu", "activity", "bar-chart-line"],
        menu_icon="cast", default_index=0,
        styles={"nav-link-selected": {"background-color": "#00b894"}}
    )

# --- 1. HALAMAN BERANDA & PANDUAN ---
if selected == "Beranda & Panduan":
    # Header Utama
    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-family: 'Segoe UI', sans-serif; color: #2d3436; font-size: 2.8rem; font-weight: 700;">
                🚗 Analisis Sentimen Transportasi Online 🛵
            </h1>
            <p style="font-family: 'Segoe UI', sans-serif; color: #636e72; font-size: 1.2rem; margin-top: 5px;">
                Komparasi Algoritma <b>Support Vector Machine (SVM)</b> dan <b>Naive Bayes</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Kotak Penjelasan Analisis Sentimen
    st.markdown("""
        <div style="font-family: 'Segoe UI', sans-serif; line-height: 1.8; color: #2d3436; background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #e6e9ef; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 35px;">
            <p style="text-align: justify; margin-bottom: 15px;">
                <b>Analisis Sentimen</b> (atau <b>Opinion Mining</b>) adalah teknik pemrosesan bahasa alami (NLP) 
                untuk mengidentifikasi, mengekstraksi, dan mempelajari informasi subjektif dalam teks. 
                Dalam penelitian ini, sistem akan mengklasifikasikan ulasan pengguna menjadi dua kategori:
            </p>
            <div style="padding-left: 5px;">
                <p style="margin-bottom: 12px;">
                    <span style="color: #00b894; font-weight: bold; font-size: 1.1rem;">● Sentimen Positif:</span><br>
                    <span style="color: #2d3436;">Ulasan yang berisi kepuasan atau pujian pelanggan terhadap layanan transportasi online.</span>
                </p>
                <p style="margin-bottom: 0;">
                    <span style="color: #d63031; font-weight: bold; font-size: 1.1rem;">● Sentimen Negatif:</span><br>
                    <span style="color: #2d3436;">Ulasan yang berisi keluhan atau ketidakpuasan pelanggan terhadap layanan transportasi online.</span>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Bagian Alur Kerja Sistem (3 Kolom Biru)
    st.markdown("<h3 style='font-family: \"Segoe UI\", sans-serif; color: #2d3436; margin-bottom: 15px;'>🛠️ Alur Kerja Sistem</h3>", unsafe_allow_html=True)
    col_step1, col_step2, col_step3 = st.columns(3)
    
    with col_step1:
        st.markdown("""
            <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; min-height: 160px; border-left: 4px solid #1e88e5; font-family: 'Segoe UI', sans-serif;">
                <b style="color: #1e88e5; font-size: 1rem;">1. Pengumpulan Data</b><br>
                <p style="color: #2d3436; font-size: 0.9rem; margin-top: 8px; line-height: 1.5;">Scraping ulasan real-time dari Google Play Store (Gojek, Grab, Maxim, inDrive).</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_step2:
        st.markdown("""
            <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; min-height: 160px; border-left: 4px solid #1e88e5; font-family: 'Segoe UI', sans-serif;">
                <b style="color: #1e88e5; font-size: 1rem;">2. Preprocessing</b><br>
                <p style="color: #2d3436; font-size: 0.9rem; margin-top: 8px; line-height: 1.5;">Pembersihan teks menggunakan library <b>Sastrawi</b> untuk mendapatkan kata dasar (Stemming).</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_step3:
        st.markdown("""
            <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; min-height: 160px; border-left: 4px solid #1e88e5; font-family: 'Segoe UI', sans-serif;">
                <b style="color: #1e88e5; font-size: 1rem;">3. Klasifikasi & Evaluasi</b><br>
                <p style="color: #2d3436; font-size: 0.9rem; margin-top: 8px; line-height: 1.5;">Membandingkan performa akurasi antara algoritma <b>SVM</b> dan <b>Naive Bayes</b>.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border: 0; border-top: 1px solid #e6e9ef;'><br>", unsafe_allow_html=True)

    # Bagian Bawah: Tentang Penelitian & Mulai Sekarang (Side-by-Side)
    col_bottom_left, col_bottom_right = st.columns([1.2, 1])

    with col_bottom_left:
        st.markdown("<h3 style='font-family: \"Segoe UI\", sans-serif; color: #2d3436; margin-bottom: 15px;'>📝 Tentang Penelitian</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <p style="font-family: 'Segoe UI', sans-serif; text-align: justify; line-height: 1.7; color: #2d3436;">
                Aplikasi ini dikembangkan oleh <b>Marco Hadi Surya</b> sebagai bagian dari penelitian tugas akhir (skripsi). 
                Tujuannya adalah untuk mengukur tingkat kepuasan dan respons pelanggan pada aplikasi transportasi online 
                (Gojek, Grab, Maxim, inDrive) dengan memanfaatkan pendekatan Machine Learning guna memperoleh 
                hasil klasifikasi sentimen yang objektif dan akurat.
            </p>
        """, unsafe_allow_html=True)
        
        # Expander Alur Preprocessing di bawah teks deskripsi
        with st.expander("📖 Lihat Alur Preprocessing Lengkap"):
            st.markdown("""
                <div style="font-family: 'Segoe UI', sans-serif; font-size: 0.95rem; line-height: 1.6; color: #2d3436;">
                    Proses pengolahan teks pada sistem ini meliputi:
                    <ul style="margin-top: 5px; padding-left: 20px;">
                        <li><b>Case Folding & Cleaning:</b> Mengubah huruf menjadi kecil serta menghapus angka, simbol, dan karakter unik.</li>
                        <li><b>Tokenizing:</b> Memotong string kalimat menjadi kumpulan token kata individu.</li>
                        <li><b>Stopword Removal:</b> Membuang kata dasar umum yang tidak membawa nilai informasi sentimen (misal: 'yang', 'di', 'dan').</li>
                        <li><b>Stemming (Sastrawi):</b> Mengembalikan kata berimbuhan menjadi kata dasar murninya.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    with col_bottom_right:
        st.markdown("<h3 style='font-family: \"Segoe UI\", sans-serif; color: #2d3436; margin-bottom: 15px;'>🚀 Mulai Sekarang</h3>", unsafe_allow_html=True)
        
        # Menampilkan langkah-langkah hijau menggunakan container html agar font-family nya sama
        langkah_langkah = [
            "1. Pilih menu <b>Input & Scraping Data</b> di sidebar.",
            "2. Ambil data via link Play Store atau unggah file CSV lokal.",
            "3. Jalankan pembersihan bertahap di menu <b>Pembersihan Data</b>.",
            "4. Analisis dan amati perbandingan akurasi di menu <b>Modeling & Evaluation</b>."
        ]
        
        for langkah in langkah_langkah:
            st.markdown(f"""
                <div style="background-color: #eef9f4; color: #1e7e4c; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px; font-family: 'Segoe UI', sans-serif; font-size: 0.95rem; border: 1px solid #c2ebd5;">
                    {langkah}
                </div>
            """, unsafe_allow_html=True)

# --- 2. HALAMAN INPUT DATA (FIXED AUTOMATED & PARALLEL PIPELINE) ---
elif selected == "Input & Scraping Data":
    st.header("📥 Input & Scraping Data")
    tab1, tab2 = st.tabs(["🌐 Scraping Play Store", "📁 Multi-Upload File"])
    
    # 📌 DEFINISI PROSES PARALEL UNTUK KEDUA TAB
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    kamus_stemming = {}

    def stem_per_baris(tokens):
        tokens_baru = []
        for kata in tokens:
            if kata in kamus_stemming:
                tokens_baru.append(kamus_stemming[kata])
            else:
                hasil_stem = stemmer.stem(kata)
                kamus_stemming[kata] = hasil_stem
                tokens_baru.append(hasil_stem)
        return tokens_baru

    jumlah_core = os.cpu_count() or 4

    # =========================================================================
    # === TAB 1: SCRAPING PLAY STORE (FIXED VARIABEL df_scraped) ===
    # =========================================================================
    with tab1:
        app_id = st.text_input("Masukkan ID Aplikasi (Contoh: com.gojek.app):", "com.gojek.app")
        limit = st.number_input("Jumlah ulasan yang ingin diambil:", min_value=10, value=100, step=50)
        
        if st.button("🚀 Mulai Scraping & Proses Otomatis"):
            with st.spinner("Sistem sedang menjalankan pipeline otomatis (Scraping ➡️ Preprocessing ➡️ Modeling)... Mohon tunggu"):
                try:
                    result, _ = reviews(app_id, lang='id', country='id', sort=Sort.NEWEST, count=limit)
                    if result:
                        df_scraped = pd.DataFrame(result)
                        st.session_state['raw_data'] = df_scraped
                        st.toast("🌐 Tahap 1: Sukses mengambil data dari Play Store!")
                        
                        # Preprocessing NLP (df_scraped)
                        def clean_text(text):
                            text = str(text).lower()
                            text = re.sub(r'[^a-zA-Z\s]', '', text)
                            return text.strip()
                            
                        df_scraped['Case_Folding_&_Cleaning'] = df_scraped['content'].apply(clean_text)
                        df_scraped['Tokenizing'] = df_scraped['Case_Folding_&_Cleaning'].apply(word_tokenize)
                        stop_words = set(stopwords.words('indonesian'))
                        df_scraped['Filtering'] = df_scraped['Tokenizing'].apply(lambda tokens: [w for w in tokens if w not in stop_words])
                        
                        # Jalankan Parallel Stemming untuk df_scraped
                        with ThreadPoolExecutor(max_workers=jumlah_core) as executor:
                            df_scraped['Stemming'] = list(executor.map(stem_per_baris, df_scraped['Filtering']))
                        df_scraped['text_clean'] = df_scraped['Stemming'].apply(lambda tokens: ' '.join(tokens))
                        
                        # Pelabelan Otomatis (Eksklusi Neutral)
                        def pelabelan_otomatis(score):
                            if score >= 4: return 'Positif'
                            elif score <= 2: return 'Negatif'
                            return None
                        df_scraped['Sentiment'] = df_scraped['score'].apply(pelabelan_otomatis)
                        df_scraped = df_scraped.dropna(subset=['Sentiment'])
                        
                        # Modeling Otomatis SVM & NB
                        tfidf = TfidfVectorizer(max_features=1000)
                        X = tfidf.fit_transform(df_scraped['text_clean'].astype(str))
                        y = df_scraped['Sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        model_svm = SVC(kernel='linear', probability=True)
                        model_svm.fit(X_train, y_train)
                        preds_svm = model_svm.predict(X_test)
                        st.session_state['svm_metrics'] = {
                            "acc": accuracy_score(y_test, preds_svm),
                            "pre": precision_score(y_test, preds_svm, pos_label='Positif'),
                            "rec": recall_score(y_test, preds_svm, pos_label='Positif'),
                            "f1": f1_score(y_test, preds_svm, pos_label='Positif')
                        }
                        
                        model_nb = MultinomialNB()
                        model_nb.fit(X_train, y_train)
                        preds_nb = model_nb.predict(X_test)
                        st.session_state['nb_metrics'] = {
                            "acc": accuracy_score(y_test, preds_nb),
                            "pre": precision_score(y_test, preds_nb, pos_label='Positif'),
                            "rec": recall_score(y_test, preds_nb, pos_label='Positif'),
                            "f1": f1_score(y_test, preds_nb, pos_label='Positif')
                        }
                        
                        tfidf_all = tfidf.transform(df_scraped['text_clean'].astype(str))
                        df_scraped['SVM_Decision_Score'] = model_svm.decision_function(tfidf_all)
                        st.session_state['processed_data'] = df_scraped
                        
                        st.toast("🧪 Tahap 3: Pelatihan Model Klasifikasi Selesai!")
                        st.success(f"🎉 Sukses Besar! {len(df_scraped)} ulasan berhasil diproses secara otomatis!")
                    else:
                        st.error("Tidak ada ulasan yang ditemukan.")
                except Exception as e:
                    st.error(f"Gagal memproses pipeline otomatis: {e}")

    # =========================================================================
    # === TAB 2: MULTI-UPLOAD FILE (MENGGUNAKAN df_gabungan) ===
    # =========================================================================
    with tab2:
        st.write("📁 *Upload Dataset Multi-Aplikasi (Gabung ➡️ Preprocessing ➡️ Modeling Otomatis)*")
        uploaded_files = st.file_uploader("Upload satu atau lebih file CSV/Excel:", type=['csv', 'xlsx'], accept_multiple_files=True)
        
        if uploaded_files:
            with st.spinner("Sistem sedang memproses seluruh file upload (Gabung ➡️ Preprocessing ➡️ Modeling)... Mohon tunggu"):
                try:
                    list_df = []
                    for f in uploaded_files:
                        df_temp = pd.read_csv(f, sep=None, engine='python') if f.name.endswith('.csv') else pd.read_excel(f)
                        nama_app = f.name.replace('.csv', '').replace('.xlsx', '').upper()
                        df_temp['Nama_Aplikasi'] = nama_app
                        list_df.append(df_temp)
                        
                    df_gabungan = pd.concat(list_df, ignore_index=True)
                    st.session_state['raw_data'] = df_gabungan
                    st.toast("📁 Tahap 1: Sukses menggabungkan file upload!")
                    
                    kolom_teks = next((k for k in ['content', 'text', 'ulasan', 'review'] if k in df_gabungan.columns), None)
                    kolom_skor = next((k for k in ['score', 'rating', 'star', 'bintang'] if k in df_gabungan.columns), None)
                    
                    if kolom_teks is None or kolom_skor is None:
                        st.error("⚠️ File harus memiliki kolom ulasan (misal: 'content') dan kolom rating bintang (misal: 'score').")
                    else:
                        def clean_text(text):
                            text = str(text).lower()
                            text = re.sub(r'[^a-zA-Z\s]', '', text)
                            return text.strip()
                            
                        df_gabungan['Case_Folding_&_Cleaning'] = df_gabungan[kolom_teks].apply(clean_text)
                        df_gabungan['Tokenizing'] = df_gabungan['Case_Folding_&_Cleaning'].apply(word_tokenize)
                        stop_words = set(stopwords.words('indonesian'))
                        df_gabungan['Filtering'] = df_gabungan['Tokenizing'].apply(lambda tokens: [w for w in tokens if w not in stop_words])
                        
                        # Jalankan Parallel Stemming untuk df_gabungan
                        with ThreadPoolExecutor(max_workers=jumlah_core) as executor:
                            df_gabungan['Stemming'] = list(executor.map(stem_per_baris, df_gabungan['Filtering']))
                        df_gabungan['text_clean'] = df_gabungan['Stemming'].apply(lambda tokens: ' '.join(tokens))
                        
                        def pelabelan_otomatis(score):
                            if score >= 4: return 'Positif'
                            elif score <= 2: return 'Negatif'
                            return None
                        df_gabungan['Sentiment'] = df_gabungan[kolom_skor].apply(pelabelan_otomatis)
                        df_gabungan = df_gabungan.dropna(subset=['Sentiment'])
                        
                        st.session_state['processed_data'] = df_gabungan
                        st.toast("🧹 Tahap 2: Preprocessing NLP Selesai!")
                        
                        tfidf = TfidfVectorizer(max_features=1000)
                        X = tfidf.fit_transform(df_gabungan['text_clean'].astype(str))
                        y = df_gabungan['Sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        model_svm = SVC(kernel='linear', probability=True)
                        model_svm.fit(X_train, y_train)
                        preds_svm = model_svm.predict(X_test)
                        st.session_state['svm_metrics'] = {
                            "acc": accuracy_score(y_test, preds_svm),
                            "pre": precision_score(y_test, preds_svm, pos_label='Positif'),
                            "rec": recall_score(y_test, preds_svm, pos_label='Positif'),
                            "f1": f1_score(y_test, preds_svm, pos_label='Positif')
                        }
                        
                        model_nb = MultinomialNB()
                        model_nb.fit(X_train, y_train)
                        preds_nb = model_nb.predict(X_test)
                        st.session_state['nb_metrics'] = {
                            "acc": accuracy_score(y_test, preds_nb),
                            "pre": precision_score(y_test, preds_nb, pos_label='Positif'),
                            "rec": recall_score(y_test, preds_nb, pos_label='Positif'),
                            "f1": f1_score(y_test, preds_nb, pos_label='Positif')
                        }
                        
                        tfidf_all = tfidf.transform(df_gabungan['text_clean'].astype(str))
                        df_gabungan['SVM_Decision_Score'] = model_svm.decision_function(tfidf_all)
                        st.session_state['processed_data'] = df_gabungan
                        
                        st.toast("🧪 Tahap 3: Pelatihan Model Selesai!")
                        st.success(f"🎉 Sukses! Berhasil memproses gabungan file dengan total {len(df_gabungan)} data ulasan bersih.")
                        
                except Exception as e:
                    st.error(f"Gagal memproses pipeline otomatis file upload: {e}")

    # --- PANEL PREVIEW & DOWNLOAD ---
    if 'raw_data' in st.session_state:
        df_mentah = st.session_state['raw_data']
        st.markdown("---")
        st.subheader(f"📋 Preview Data Mentah (Menampilkan 20 dari {len(df_mentah)} total baris)")
        st.dataframe(df_mentah.head(20), use_container_width=True)
        
        csv_mentah = df_mentah.to_csv(index=False, sep=";").encode('utf-8-sig')
        st.download_button(
            label=f"⬇️ Download Seluruh Data Mentah ({len(df_mentah)} Baris)", 
            data=csv_mentah, 
            file_name=f"data_mentah_scraping_{len(df_mentah)}.csv", 
            mime="text/csv"
        )

# --- 3. HALAMAN PEMBERSIHAN ---
elif selected == "Pembersihan Data":
    st.header("🧹 Preprocessing Data (NLP Pipeline)")
    if 'raw_data' not in st.session_state:
        st.warning("Silakan input data terlebih dahulu di menu 'Input & Scraping Data'.")
    else:
        df = st.session_state['raw_data'].copy()
        kolom_teks = st.selectbox("Pilih kolom ulasan:", df.columns)
        
        if st.button("Mulai Pembersihan"):
            progress_text = "Memulai tahapan Preprocessing..."
            my_bar = st.progress(0, text=progress_text)
            
            # --- TAHAP 1: CASE FOLDING & CLEANING ---
            my_bar.progress(20, text="Tahap 1: Case Folding & Cleaning...")
            def clean_text(text):
                text = str(text).lower()
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                return text.strip()
            df['Case_Folding_&_Cleaning'] = df[kolom_teks].apply(clean_text)
            
            # --- TAHAP 2: TOKENIZING ---
            my_bar.progress(40, text="Tahap 2: Tokenizing...")
            df['Tokenizing'] = df['Case_Folding_&_Cleaning'].apply(word_tokenize)
            
            # --- TAHAP 3: FILTERING (STOPWORD REMOVAL) ---
            my_bar.progress(60, text="Tahap 3: Filtering (Stopwords)...")
            stop_words = set(stopwords.words('indonesian'))
            df['Filtering'] = df['Tokenizing'].apply(lambda tokens: [w for w in tokens if w not in stop_words])
            
            # --- TAHAP 4: STEMMING FAST VERSION (CACHED SASTRAWI) ---
            my_bar.progress(80, text="Tahap 4: Stemming Sastrawi...")
            
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            
            # 🚀 KUNCI KECEPATAN: Buat kamus kosong untuk menyimpan kata yang sudah di-stem
            kamus_stemming = {}
            
            def stem_cepat(tokens):
                tokens_baru = []
                for kata in tokens:
                    if kata in kamus_stemming:
                        tokens_baru.append(kamus_stemming[kata])
                    else:
                        hasil_stem = stemmer.stem(kata)
                        kamus_stemming[kata] = hasil_stem
                        tokens_baru.append(hasil_stem)
                return tokens_baru

            # Jalankan fungsi stem cepat ke seluruh baris data
            df['Stemming'] = df['Filtering'].apply(stem_cepat)
            
            # Satukan kembali token menjadi string kalimat bersih
            df['text_clean'] = df['Stemming'].apply(lambda tokens: ' '.join(tokens))
            
            my_bar.progress(100, text="Semua tahap preprocessing selesai!")
            st.session_state['processed_data'] = df
            st.success(f"✅ Preprocessing {len(df)} data selesai!")

        # Tampilkan tabel per tahap
        if 'processed_data' in st.session_state:
            st.subheader("Hasil Preprocessing per Tahapan (Preview 20 Data)")
            kolom_tampil = [kolom_teks, 'Case_Folding_&_Cleaning', 'Tokenizing', 'Filtering', 'Stemming']
            st.dataframe(st.session_state['processed_data'][kolom_tampil].head(20), use_container_width=True)
            
            csv_bersih = st.session_state['processed_data'].to_csv(index=False).encode('utf-8')
            st.download_button(label="⬇️ Download Data Bersih (CSV)", data=csv_bersih, file_name="data_bersih_skripsi.csv", mime="text/csv")

# --- 4. HALAMAN ANALISIS SVM ---
elif selected == "Support Vector Machine (SVM)":
    st.header("🛡️ Support Vector Machine (SVM)")
    
    if 'processed_data' not in st.session_state:
        st.warning("⚠️ Selesaikan tahap Pembersihan Data terlebih dahulu.")
    else:
        # Gunakan data dari session state
        df_svm = st.session_state['processed_data'].copy()
        
        # 🚀 SOLUSI KUNCI: Pastikan pelabelan sentimen berjalan sebelum data di-split
        if 'score' in df_svm.columns and 'Sentiment' not in df_svm.columns:
            def pelabelan_otomatis(score):
                if score >= 4: return 'Positif'
                elif score <= 2: return 'Negatif'
                return None
            df_svm['Sentiment'] = df_svm['score'].apply(pelabelan_otomatis)
            df_svm = df_svm.dropna(subset=['Sentiment'])
            st.session_state['processed_data'] = df_svm

        if st.button("🚀 Jalankan Pengujian SVM"):
            with st.spinner("Mencari Hyperplane Optimal..."):
                tfidf = TfidfVectorizer(max_features=1000)
                X = tfidf.fit_transform(df_svm['text_clean'].astype(str))
                y = df_svm['Sentiment'] # Sekarang kolom ini dijamin ada!
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model_svm = SVC(kernel='linear', probability=True)
                model_svm.fit(X_train, y_train)
                preds = model_svm.predict(X_test)
                
                st.session_state['svm_metrics'] = {
                    "acc": accuracy_score(y_test, preds),
                    "pre": precision_score(y_test, preds, pos_label='Positif'),
                    "rec": recall_score(y_test, preds, pos_label='Positif'),
                    "f1": f1_score(y_test, preds, pos_label='Positif')
                }
                
                # Update label prediksi ke data utama untuk dashboard
                tfidf_all = tfidf.transform(df_svm['text_clean'].astype(str))
                df_svm['Sentiment'] = model_svm.predict(tfidf_all)
                st.session_state['processed_data'] = df_svm
                
                st.success("Analisis SVM Selesai!")

        if 'svm_metrics' in st.session_state:
            m = st.session_state['svm_metrics']
            st.subheader("📋 Metrik Evaluasi SVM")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{m['acc']*100:.2f}%")
            col2.metric("Precision", f"{m['pre']*100:.2f}%")
            col3.metric("Recall", f"{m['rec']*100:.2f}%")
            col4.metric("F1-Score", f"{m['f1']*100:.2f}%")
            st.info("💡 **Analisis:** SVM bekerja dengan mencari garis pembatas (Hyperplane) terbaik untuk memisahkan data.")

# --- 5. HALAMAN ANALISIS NAIVE BAYES ---
elif selected == "Naive Bayes":
    st.header("📊 Naive Bayes (NB)")
    
    if 'processed_data' not in st.session_state:
        st.warning("⚠️ Selesaikan tahap Pembersihan Data terlebih dahulu.")
    else:
        df_nb = st.session_state['processed_data'].copy()
        
        # 🚀 SOLUSI KUNCI: Pastikan pelabelan sentimen berjalan sebelum data di-split
        if 'score' in df_nb.columns and 'Sentiment' not in df_nb.columns:
            def pelabelan_otomatis(score):
                if score >= 4: return 'Positif'
                elif score <= 2: return 'Negatif'
                return None
            df_nb['Sentiment'] = df_nb['score'].apply(pelabelan_otomatis)
            df_nb = df_nb.dropna(subset=['Sentiment'])
            st.session_state['processed_data'] = df_nb

        if st.button("🚀 Jalankan Pengujian Naive Bayes"):
            with st.spinner("Menghitung Probabilitas Kata..."):
                tfidf = TfidfVectorizer(max_features=1000)
                X = tfidf.fit_transform(df_nb['text_clean'].astype(str))
                y = df_nb['Sentiment'] # Sekarang kolom ini dijamin ada!
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model_nb = MultinomialNB()
                model_nb.fit(X_train, y_train)
                preds = model_nb.predict(X_test)
                
                st.session_state['nb_metrics'] = {
                    "acc": accuracy_score(y_test, preds),
                    "pre": precision_score(y_test, preds, pos_label='Positif'),
                    "rec": recall_score(y_test, preds, pos_label='Positif'),
                    "f1": f1_score(y_test, preds, pos_label='Positif')
                }
                st.success("Analisis Naive Bayes Selesai!")

        if 'nb_metrics' in st.session_state:
            m = st.session_state['nb_metrics']
            st.subheader("📋 Metrik Evaluasi Naive Bayes")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{m['acc']*100:.2f}%")
            col2.metric("Precision", f"{m['pre']*100:.2f}%")
            col3.metric("Recall", f"{m['rec']*100:.2f}%")
            col4.metric("F1-Score", f"{m['f1']*100:.2f}%")
            st.info("💡 **Analisis:** Naive Bayes menghitung probabilitas kemunculan setiap kata untuk menentukan sentimen.")

# --- 6. HALAMAN DASHBOARD ---
elif selected == "Dashboard & Visualisasi":
    st.header("📊 Dashboard & Visualisasi Analisis Sentimen")

    if 'processed_data' in st.session_state and 'svm_metrics' in st.session_state:
        df_vis = st.session_state['processed_data']
        
        # =========================================================================
        # ⚔️ BAGIAN 1: KOMPARASI PERFORMA GLOBAL (SVM VS NAIVE BAYES)
        # =========================================================================
        st.subheader("⚔️ Komparasi Performa: SVM vs Naive Bayes")
        
        svm = st.session_state['svm_metrics']
        nb = st.session_state['nb_metrics']

        # Menampilkan kartu metrik berdampingan
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            diff_acc = (svm['acc'] - nb['acc']) * 100
            st.metric("Accuracy (SVM)", f"{svm['acc']*100:.2f}%", f"{diff_acc:+.2f}% vs NB")
        with m2:
            st.metric("Precision (SVM)", f"{svm['pre']*100:.2f}%")
        with m3:
            st.metric("Recall (SVM)", f"{svm['rec']*100:.2f}%")
        with m4:
            st.metric("F1-Score (SVM)", f"{svm['f1']*100:.2f}%")

        # Grafik Batang Perbandingan
        df_metrics = pd.DataFrame({
            'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score'] * 2,
            'Nilai': [svm['acc'], svm['pre'], svm['rec'], svm['f1'], nb['acc'], nb['pre'], nb['rec'], nb['f1']],
            'Algoritma': ['Support Vector Machine'] * 4 + ['Naive Bayes'] * 4
        })
        
        fig_comp = px.bar(df_metrics, x='Metrik', y='Nilai', color='Algoritma', barmode='group',
                          color_discrete_map={'Support Vector Machine':'#2ecc71', 'Naive Bayes':'#3498db'},
                          text_auto='.2%', title="Perbandingan Metrik Evaluasi Model")
        fig_comp.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_comp, use_container_width=True)

        winner = "Support Vector Machine (SVM)" if svm['acc'] > nb['acc'] else "Naive Bayes"
        st.info(f"💡 **Kesimpulan Pengujian:** Berdasarkan metrik akurasi, algoritma **{winner}** memberikan hasil klasifikasi yang lebih optimal untuk dataset ulasan ini.")

        # =========================================================================
        # 📈 BAGIAN 2: BREAKDOWN CHART & WORDCLOUD PER APLIKASI (FIXED SAFETY CHECK)
        # =========================================================================
        st.markdown("---")
        st.subheader("📱 Breakdown Analisis Sentimen & Wordcloud Per Aplikasi")
        
        # 🚀 KUNCI PENYELAMAT: Cek apakah kolom 'Nama_Aplikasi' benar-benar ada di data
        if 'Nama_Aplikasi' in df_vis.columns:
            st.write("Berikut adalah visualisasi distribusi sentimen dan kata kunci populer yang dipisah untuk masing-masing aplikasi transportasi online:")
            
            # Ambil daftar aplikasi unik secara otomatis (Gojek, Grab, Maxim, inDrive)
            apps_terdeteksi = sorted(list(df_vis['Nama_Aplikasi'].unique()))

            # Tampilkan grafik berdampingan per aplikasi dalam wadah expander
            for nama_app in apps_terdeteksi:
                df_app = df_vis[df_vis['Nama_Aplikasi'] == nama_app]
                
                with st.expander(f"🟢 ANALISIS UNTUK APLIKASI: {nama_app} ({len(df_app)} Data)", expanded=True):
                    col_pie, col_wc = st.columns([1, 1])
                    
                    # 1. Pie Chart (Kiri)
                    with col_pie:
                        if not df_app.empty:
                            fig_pie_app = px.pie(
                                df_app, names='Sentiment', color='Sentiment',
                                color_discrete_map={'Positif':'#2ecc71', 'Negatif':'#e74c3c'},
                                hole=0.4, title=f"Persentase Sentimen {nama_app}"
                            )
                            fig_pie_app.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_pie_app, use_container_width=True)
                        else:
                            st.write("Tidak ada data ulasan untuk aplikasi ini.")
                    
                    # 2. Wordcloud (Kanan)
                    with col_wc:
                        text_app = " ".join(df_app['text_clean'].astype(str))
                        if text_app.strip():
                            wordcloud_app = WordCloud(
                                width=600, height=350, 
                                background_color='white', 
                                colormap='plasma',
                                max_words=50
                            ).generate(text_app)
                            
                            st.image(
                                wordcloud_app.to_array(), 
                                caption=f"WordCloud Kata Kunci Terpopuler pada Aplikasi {nama_app}", 
                                use_container_width=True
                            )
                        else:
                            st.write("Teks ulasan bersih tidak mencukupi untuk membuat Wordcloud.")
        
        else:
            # 💡 JIKA BELUM ADA KOLOM (Data Tunggal / Scraping Versi Lama), TAMPILKAN SECARA GLOBAL SAJA
            st.write("Berikut adalah visualisasi distribusi sentimen dan kata kunci populer secara menyeluruh (Global Dataset):")
            
            col_pie, col_wc = st.columns([1, 1])
            with col_pie:
                fig_pie_global = px.pie(
                    df_vis, names='Sentiment', color='Sentiment',
                    color_discrete_map={'Positif':'#2ecc71', 'Negatif':'#e74c3c'},
                    hole=0.4, title="Persentase Sentimen Global"
                )
                fig_pie_global.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie_global, use_container_width=True)
                
            with col_wc:
                text_global = " ".join(df_vis['text_clean'].astype(str))
                if text_global.strip():
                    wordcloud_global = WordCloud(width=600, height=350, background_color='white', colormap='plasma', max_words=50).generate(text_global)
                    st.image(wordcloud_global.to_array(), caption="WordCloud Kata Kunci Global", use_container_width=True)
                else:
                    st.write("Teks tidak mencukupi.")
    else:
        st.warning("⚠️ Data belum diproses atau dimuat ke sistem. Silakan lakukan proses di halaman 'Input & Scraping Data' terlebih dahulu.")