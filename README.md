# Analisis Sentimen Ulasan Mobile Legends Menggunakan Word2Vec dan LSTM

## 📌 Tentang Proyek

Proyek ini mengimplementasikan **Natural Language Processing (NLP)** dan *Deep Learning* untuk mengklasifikasikan sentimen ulasan pengguna game Mobile Legends menjadi dua kategori: **positif** dan **negatif**.

Word2Vec digunakan untuk mengubah kata-kata menjadi representasi vektor numerik, sedangkan Long Short-Term Memory (LSTM) digunakan untuk mempelajari pola urutan kata dan memprediksi sentimen dari ulasan pengguna.

Proyek ini bertujuan untuk mengeksplorasi penerapan Word2Vec dan LSTM dalam analisis sentimen teks serta mengevaluasi performa model menggunakan data pengujian.

## 🎯 Tujuan

- Mengolah dan membersihkan teks ulasan pengguna.
- Mengklasifikasikan ulasan ke dalam sentimen positif dan negatif berdasarkan skor rating.
- Membangun representasi kata menggunakan Word2Vec.
- Melatih model LSTM untuk klasifikasi sentimen.
- Mengevaluasi model menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.
- Melakukan prediksi sentimen pada ulasan baru.

## 📂 Dataset

Dataset yang digunakan berisi **100.000 ulasan pengguna Mobile Legends**.

Kolom yang digunakan dalam implementasi meliputi:

| Kolom | Deskripsi |
|---|---|
| `userName` | Nama pengguna |
| `score` | Rating ulasan |
| `at` | Waktu ulasan dibuat |
| `content` | Teks ulasan pengguna |

Dataset diproses untuk mempertahankan kolom yang relevan, kemudian diurutkan berdasarkan waktu ulasan.


## 🛠️ Teknologi dan Library

- **Python** — bahasa pemrograman.
- **Pandas** — manipulasi dan pengolahan data.
- **NLTK** — tokenisasi dan penghapusan stopwords.
- **Gensim** — pelatihan model Word2Vec.
- **TensorFlow / Keras** — pembangunan dan pelatihan model LSTM.
- **Scikit-learn** — pembagian dataset dan evaluasi model.
- **Matplotlib, Seaborn, Plotly** — visualisasi hasil analisis.
- **WordCloud** — visualisasi kata yang sering muncul.

## ⚙️ Metodologi

### 1. Data Preparation

Data ulasan dibaca dari file CSV. Kolom yang diperlukan dipilih dan data diurutkan berdasarkan waktu ulasan.

### 2. Sentiment Labeling

Pada tahap pemodelan, rating dikonversi menjadi label numerik:

| Rating | Label | Nilai |
|---|---|---:|
| 1–2 | Negatif | 0 |
|  3  | Netral / Tidak digunakan | - |
| 4–5 | Positif | 1 |

Label tersebut digunakan sebagai target dalam proses pelatihan model.

### 3. Text Preprocessing

Tahapan preprocessing yang diterapkan meliputi:

1. **Cleaning** — menghapus karakter selain huruf dan spasi menggunakan regular expression.
2. **Case folding** — mengubah seluruh teks menjadi huruf kecil.
3. **Stopword removal** — menghapus stopwords bahasa Indonesia menggunakan NLTK.
4. **Tokenization** — memecah teks menjadi token kata.

### 4. Word Embedding dengan Word2Vec

Word2Vec dilatih menggunakan teks hasil preprocessing dengan konfigurasi berikut:

| Parameter | Nilai |
|---|---|
| Architecture | Skip-gram |
| Vector size | 100 |
| Window size | 5 |
| Minimum word count | 5 |
| Workers | 4 |

Vektor kata yang dihasilkan digunakan untuk membentuk embedding matrix sebagai masukan bagi model LSTM.

### 5. Sequence Preparation

Token kata dikonversi menjadi indeks berdasarkan kosakata Word2Vec. Selanjutnya, urutan kata disesuaikan menjadi panjang maksimum **100 token** menggunakan `pad_sequences`.

Dataset dibagi menjadi:

- **80% data latih**
- **20% data uji**

Pembagian dilakukan menggunakan `train_test_split` dengan `random_state=42`.

### 6. LSTM Model

Arsitektur model yang digunakan terdiri dari:

1. **Embedding Layer** — menggunakan vektor Word2Vec dengan dimensi 100. Bobot embedding dibekukan selama pelatihan.
2. **SpatialDropout1D** — dengan dropout rate 0,2.
3. **LSTM Layer** — 100 unit, dengan dropout dan recurrent dropout masing-masing 0,2.
4. **Dense Layer** — 2 unit dengan aktivasi softmax untuk klasifikasi sentimen.

Konfigurasi pelatihan:

| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |
| Batch size | 64 |
| Maximum epochs | 20 |
| Early stopping | Patience 3 |
| Validation data | Data uji |

Model menggunakan callback `EarlyStopping` untuk menghentikan pelatihan apabila validation loss tidak membaik selama beberapa epoch sesuai konfigurasi.

## 📊 Hasil Evaluasi Model

Berdasarkan output evaluasi yang tersimpan di notebook, model memperoleh hasil berikut pada **20.000 data uji**.

| Metrik | Hasil |
|---|---:|
| Accuracy | 87,83% |
| Precision — kelas positif | 91,08% |
| Recall — kelas positif | 92,25% |
| F1-score — kelas positif | 91,66% |

### Classification Report

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negatif (0) | 0,79 | 0,76 | 0,77 | 5.500 |
| Positif (1) | 0,91 | 0,92 | 0,92 | 14.500 |
| **Accuracy** | | | **0,88** | **20.000** |
| Macro average | 0,85 | 0,84 | 0,85 | 20.000 |
| Weighted average | 0,88 | 0,88 | 0,88 | 20.000 |

### Confusion Matrix

Berdasarkan hasil prediksi pada notebook:

| | Prediksi Negatif | Prediksi Positif |
|---|---:|---:|
| Aktual Negatif | 4.190 | 1.310 |
| Aktual Positif | 1.124 | 13.376 |

Hasil tersebut menunjukkan bahwa model mengklasifikasikan sebagian besar ulasan dengan benar. Namun, performa pada kelas negatif lebih rendah dibandingkan kelas positif, sehingga masih terdapat ulasan negatif yang diprediksi sebagai positif.

## 🔍 Contoh Prediksi

Notebook juga menguji model menggunakan contoh ulasan baru:

**Input:**
```text
game aneh
```

**Hasil prediksi:**
```text
Negatif
```

Hasil ini merupakan contoh prediksi dari model yang sudah dilatih, bukan jaminan bahwa semua ulasan dengan karakteristik serupa akan diklasifikasikan secara tepat.

## 🚀 Cara Menjalankan Proyek

### 1. Clone repository

```bash
git clone https://github.com/USERNAME/REPOSITORY-NAME.git
cd REPOSITORY-NAME
```

Ganti `USERNAME` dan `REPOSITORY-NAME` dengan informasi repositorimu.

### 2. Install dependencies

```bash
pip install pandas numpy nltk gensim scikit-learn tensorflow matplotlib seaborn plotly wordcloud
```

### 3. Siapkan dataset

Tempatkan dataset CSV di lokasi yang sesuai. Notebook saat ini menggunakan path Google Colab berikut:

```python
/content/mobile_legends_reviews.csv
```

Notebook juga menghasilkan file perantara:

```text
Sort_mobile_legend.csv
```

Jika dijalankan di luar Google Colab, sesuaikan path file dengan lokasi dataset di komputermu.

### 4. Jalankan notebook

Buka notebook melalui Google Colab atau Jupyter Notebook, lalu jalankan sel secara berurutan dari tahap persiapan data sampai evaluasi model.

## 📁 Struktur Repository

```text
mobile-legends-sentiment-analysis/
│
├── dataset/
│   ├── mobile_legends_reviews.csv
│   └── sort_mobile_legends.csv
│
├── models/
│   ├── sentiment_analysis_model_lstm.h5
│   └── word2vec_mobile_legends .model
│
├── notebooks/
│   └── sentiment_analysis_lstm_word2vec.ipynb
|
├── results/
│   ├── confusion_matrix.png
│   ├── distribusi_sentimen.png
│   ├── evaluasi_model.png
│   ├── training_model.png
│   └── work_cloud.png
│
└── README.md
```


## 📌 Kesimpulan

Proyek ini menunjukkan penerapan Word2Vec sebagai representasi kata dan LSTM sebagai model klasifikasi sentimen pada ulasan Mobile Legends. Berdasarkan output notebook, model mencapai akurasi 87,83% pada data uji.

Proyek ini juga menjadi sarana untuk menerapkan tahapan pengolahan data teks, pembentukan word embedding, pembangunan model deep learning, serta evaluasi performa klasifikasi.

