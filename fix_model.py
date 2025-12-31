"""
Script untuk memperbaiki model yang sudah ada dengan menambahkan probability=True
Jalankan script ini untuk memperbaiki model yang sudah di-save sebelumnya
"""

import joblib
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

print("🔧 Memperbaiki model dengan probability=True...")

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv('sentimendataset.csv')
df.columns = ['text']
df = df.dropna()
df = df[df['text'].str.strip() != '']

# Membuat label sentimen
def create_sentiment_label(text):
    text_lower = text.lower()
    positive_keywords = ['bagus', 'baik', 'puas', 'senang', 'mudah', 'cepat', 'terbantu', 
                         'mantap', 'recommend', 'suka', 'menyenangkan', 'mempermudah',
                         'terima kasih', 'terimakasih', 'helpful', 'enak']
    negative_keywords = ['buruk', 'jelek', 'kecewa', 'mengecewakan', 'susah', 'lama', 
                        'error', 'bug', 'masalah', 'tidak', 'gak', 'ga', 'payah',
                        'parah', 'jengkel', 'risih', 'repot', 'mahal', 'telat', 'batal']
    
    pos_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    neg_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    if neg_count > pos_count:
        return 'negatif'
    elif pos_count > neg_count:
        return 'positif'
    else:
        if len(text) < 50:
            return 'positif'
        return 'negatif'

df['sentiment'] = df['text'].apply(create_sentiment_label)

# Undersampling
print("⚖️ Menyeimbangkan data...")
df_positif = df[df['sentiment'] == 'positif'].copy()
n_positif = len(df_positif)
df_negatif = df[df['sentiment'] == 'negatif'].sample(n=n_positif, random_state=42)
df_balanced = pd.concat([df_positif, df_negatif], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
df = df_balanced.copy()

# Preprocessing
def case_folding(text):
    return text.lower()

def cleaning(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
additional_stopwords = ['aplikasi', 'gojek', 'gofood', 'goride', 'gocar', 'gosend', 
                        'driver', 'pesan', 'order', 'pesanan', 'gopay', 'gopinjam']

def remove_stopwords(text):
    text = stopword.remove(text)
    words = text.split()
    words = [word for word in words if word not in additional_stopwords]
    return ' '.join(words)

factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

def stemming_text(text):
    return stemmer.stem(text)

print("🔧 Preprocessing...")
df['text_casefold'] = df['text'].apply(case_folding)
df['text_cleaned'] = df['text_casefold'].apply(cleaning)
df['text_no_stopword'] = df['text_cleaned'].apply(remove_stopwords)
df['text_stemmed'] = df['text_no_stopword'].apply(stemming_text)

# Load vectorizer yang sudah ada (jika ada)
try:
    print("📥 Loading vectorizer yang sudah ada...")
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ Vectorizer loaded")
except FileNotFoundError:
    print("⚠️ Vectorizer tidak ditemukan, membuat baru...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X_tfidf = vectorizer.fit_transform(df['text_stemmed'].values)
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("✅ Vectorizer baru dibuat dan disimpan")

# TF-IDF
X = df['text_stemmed'].values
y = df['sentiment'].values
X_tfidf = vectorizer.transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load model yang sudah ada untuk melihat kernel
try:
    old_model = joblib.load('svm_model.pkl')
    kernel_type = old_model.kernel
    print(f"📥 Model lama ditemukan dengan kernel: {kernel_type}")
    print("🔄 Re-training model dengan probability=True...")
except FileNotFoundError:
    kernel_type = 'linear'
    print("⚠️ Model tidak ditemukan, membuat model baru dengan kernel linear...")

# Re-train dengan probability=True
if kernel_type == 'linear':
    model = SVC(kernel='linear', random_state=42, probability=True)
else:
    model = SVC(kernel='rbf', random_state=42, probability=True)

model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model berhasil di-re-train!")
print(f"📊 Akurasi: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🔧 Kernel: {kernel_type}")
print(f"✅ probability=True: {model.probability}")

# Simpan model yang sudah diperbaiki
joblib.dump(model, 'svm_model.pkl')
print("\n💾 Model yang sudah diperbaiki berhasil disimpan!")
print("🚀 Sekarang Anda bisa menjalankan: streamlit run app.py")

