import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import joblib
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen dengan SVM",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi preprocessing
@st.cache_resource
def load_preprocessing():
    """Load preprocessing tools"""
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    
    factory_stopword = StopWordRemoverFactory()
    stopword = factory_stopword.create_stop_word_remover()
    
    additional_stopwords = [
        'aplikasi', 'gojek', 'gofood', 'goride', 'gocar', 'gosend', 
        'driver', 'pesan', 'order', 'pesanan', 'gopay', 'gopinjam'
    ]
    
    return stemmer, stopword, additional_stopwords

# Fungsi preprocessing
def case_folding(text):
    """Mengubah semua teks menjadi lowercase"""
    return text.lower()

def cleaning(text):
    """Membersihkan teks dari angka, tanda baca, emoji"""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def remove_stopwords(text, stopword, additional_stopwords):
    """Menghapus stopword"""
    text = stopword.remove(text)
    words = text.split()
    words = [word for word in words if word not in additional_stopwords]
    return ' '.join(words)

def preprocess_text(text, stemmer, stopword, additional_stopwords):
    """Preprocessing lengkap"""
    text = case_folding(text)
    text = cleaning(text)
    text = remove_stopwords(text, stopword, additional_stopwords)
    text = stemmer.stem(text)
    return text

# Load model dan vectorizer
@st.cache_resource
def load_model():
    """Load trained model dan vectorizer"""
    try:
        # Coba load dari file pickle/joblib
        # Jika belum ada, akan dibuat dari training
        vectorizer = joblib.load('vectorizer.pkl')
        model = joblib.load('svm_model.pkl')
        
        # Cek apakah model memiliki predict_proba
        if not hasattr(model, 'predict_proba') or not model.probability:
            st.error("""
            ⚠️ **Model tidak memiliki probability=True!**
            
            Model yang di-load tidak bisa menggunakan `predict_proba`.
            Silakan jalankan script berikut untuk memperbaiki model:
            
            ```bash
            python fix_model.py
            ```
            
            Atau re-train model di notebook dengan menambahkan `probability=True`:
            ```python
            svm_linear = SVC(kernel='linear', random_state=42, probability=True)
            ```
            """)
            return None, None
        
        return vectorizer, model
    except FileNotFoundError:
        st.error("⚠️ Model belum di-train. Silakan jalankan script training terlebih dahulu.")
        return None, None

# Header
st.markdown('<h1 class="main-header">Analisis Sentimen dengan SVM</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Informasi")
    st.info("""
    **Aplikasi Analisis Sentimen**
    
    Aplikasi ini menggunakan model SVM untuk menganalisis sentimen teks dalam bahasa Indonesia.
    
    **Fitur:**
    - Preprocessing otomatis
    - Prediksi sentimen (Positif/Negatif)
    - Visualisasi hasil
    """)
    
    st.markdown("---")
    st.header("Pengaturan")
    show_preprocessing = st.checkbox("Tampilkan proses preprocessing", value=False)

# Load preprocessing tools
stemmer, stopword, additional_stopwords = load_preprocessing()

# Load model
vectorizer, model = load_model()

# Main content
if model is None or vectorizer is None:
    st.warning("""
    **Model belum tersedia.**
    
    Untuk menggunakan aplikasi ini, Anda perlu:
    1. Menjalankan script training untuk membuat model
    2. Menyimpan model dan vectorizer menggunakan joblib/pickle
    
    Apakah Anda ingin saya buatkan script untuk training dan menyimpan model?
    """)
else:
    # Tab untuk input teks
    tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Sentimen", "📊 Analisis Batch", "ℹ️ Tentang"])
    
    with tab1:
        st.header("Input Teks untuk Analisis")
        
        # Input text
        text_input = st.text_area(
            "Masukkan teks yang ingin dianalisis:",
            height=150,
            placeholder="Contoh: Aplikasi ini sangat membantu dan mudah digunakan..."
        )
        
        if st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("Sedang menganalisis..."):
                    # Preprocessing
                    processed_text = preprocess_text(text_input, stemmer, stopword, additional_stopwords)
                    
                    # Transform dengan vectorizer
                    text_vectorized = vectorizer.transform([processed_text])
                    
                    # Prediksi
                    prediction = model.predict(text_vectorized)[0]
                    probability = model.predict_proba(text_vectorized)[0]
                    
                    # Tampilkan hasil preprocessing jika diminta
                    if show_preprocessing:
                        with st.expander("📝 Proses Preprocessing", expanded=True):
                            st.write("**Teks Asli:**")
                            st.code(text_input, language=None)
                            st.write("**Setelah Preprocessing:**")
                            st.code(processed_text, language=None)
                    
                    # Tampilkan hasil
                    st.markdown("---")
                    st.subheader("🎯 Hasil Prediksi")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 'positif':
                            st.markdown(f"""
                            <div class="prediction-box positive">
                                <h2>Sentimen: POSITIF</h2>
                                <p style="font-size: 1.2rem;">Tingkat Keyakinan: <strong>{probability[1]*100:.2f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box negative">
                                <h2>Sentimen: NEGATIF</h2>
                                <p style="font-size: 1.2rem;">Tingkat Keyakinan: <strong>{probability[0]*100:.2f}%</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        # Visualisasi probabilitas
                        fig, ax = plt.subplots(figsize=(8, 6))
                        labels = ['Negatif', 'Positif']
                        colors = ['#dc3545', '#28a745']
                        bars = ax.bar(labels, probability, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                        ax.set_ylabel('Probabilitas', fontsize=12, fontweight='bold')
                        ax.set_title('Distribusi Probabilitas Sentimen', fontsize=14, fontweight='bold', pad=15)
                        ax.set_ylim([0, 1])
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        
                        # Tambahkan nilai di atas bar
                        for bar, prob in zip(bars, probability):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{prob*100:.2f}%',
                                   ha='center', va='bottom', fontsize=12, fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            else:
                st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")
    
    with tab2:
        st.header("Analisis Batch (Multiple Teks)")
        
        # Upload file atau input manual
        option = st.radio("Pilih metode input:", ["Input Manual", "Upload File CSV"])
        
        if option == "Input Manual":
            texts_input = st.text_area(
                "Masukkan beberapa teks (satu teks per baris):",
                height=200,
                placeholder="Teks 1\nTeks 2\nTeks 3..."
            )
            
            if st.button("📊 Analisis Batch", type="primary"):
                if texts_input.strip():
                    texts = [t.strip() for t in texts_input.split('\n') if t.strip()]
                    
                    results = []
                    with st.spinner(f"Sedang menganalisis {len(texts)} teks..."):
                        for i, text in enumerate(texts):
                            processed = preprocess_text(text, stemmer, stopword, additional_stopwords)
                            text_vec = vectorizer.transform([processed])
                            pred = model.predict(text_vec)[0]
                            prob = model.predict_proba(text_vec)[0]
                            
                            results.append({
                                'Teks': text[:100] + '...' if len(text) > 100 else text,
                                'Sentimen': pred,
                                'Probabilitas': f"{max(prob)*100:.2f}%"
                            })
                    
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Statistik
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Teks", len(texts))
                    with col2:
                        positif_count = sum(1 for r in results if r['Sentimen'] == 'positif')
                        st.metric("Positif", positif_count)
                    with col3:
                        negatif_count = sum(1 for r in results if r['Sentimen'] == 'negatif')
                        st.metric("Negatif", negatif_count)
                else:
                    st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")
        
        else:
            uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview data:")
                    st.dataframe(df.head())
                    
                    # Pilih kolom
                    text_column = st.selectbox("Pilih kolom yang berisi teks:", df.columns)
                    
                    if st.button("📊 Analisis Batch", type="primary"):
                        texts = df[text_column].dropna().tolist()
                        results = []
                        
                        with st.spinner(f"Sedang menganalisis {len(texts)} teks..."):
                            for text in texts:
                                processed = preprocess_text(str(text), stemmer, stopword, additional_stopwords)
                                text_vec = vectorizer.transform([processed])
                                pred = model.predict(text_vec)[0]
                                prob = model.predict_proba(text_vec)[0]
                                
                                results.append({
                                    'Teks': str(text)[:100] + '...' if len(str(text)) > 100 else str(text),
                                    'Sentimen': pred,
                                    'Probabilitas': f"{max(prob)*100:.2f}%"
                                })
                        
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Download hasil
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Hasil (CSV)",
                            data=csv,
                            file_name="hasil_analisis_sentimen.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab3:
        st.header("ℹ️ Tentang Aplikasi")
        st.markdown("""
        ### Deskripsi
        Aplikasi ini menggunakan model **Support Vector Machine (SVM)** untuk menganalisis sentimen teks dalam bahasa Indonesia.
        
        ### Teknologi yang Digunakan
        - **Model**: SVM (Support Vector Machine) dengan kernel Linear
        - **Preprocessing**: 
          - Case Folding
          - Cleaning (menghapus angka, tanda baca, emoji)
          - Stopword Removal (Sastrawi + manual)
          - Stemming (Sastrawi)
        - **Feature Extraction**: TF-IDF Vectorization
        
        ### Dataset
        Dataset yang digunakan adalah review aplikasi Gojek dengan label sentimen Positif dan Negatif.
        
        ### Cara Menggunakan
        1. Masukkan teks yang ingin dianalisis di tab "Prediksi Sentimen"
        2. Klik tombol "Analisis Sentimen"
        3. Lihat hasil prediksi dan visualisasi probabilitas
        
        ### Catatan
        - Model ini dilatih dengan data yang sudah di-balance (undersampling)
        - Akurasi model: ~89% (SVM Linear)
        - Model dapat ditingkatkan dengan lebih banyak data training
        """)

