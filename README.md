# 🔗 Association Rule Mining System — Apriori

> Implementasi algoritma Apriori untuk analisis pola pembelian, dilengkapi Sistem Rekomendasi Produk, SPK Tata Letak Rak, dan Prediksi Perilaku Pembelian berbasis data transaksi online retail.

---

## 📋 Deskripsi Proyek

Aplikasi ini dikembangkan sebagai tugas implementasi **Association Rule Mining (ARM)** menggunakan algoritma Apriori. Tidak hanya menampilkan frequent itemsets dan association rules, aplikasi ini juga mengimplementasikan ARM ke dalam tiga sistem nyata yang saling terhubung.

---

## 📂 Dataset

### Online Retail II (2009–2010)
Dataset yang digunakan adalah **Online Retail II** dari UCI Machine Learning Repository, khususnya data periode **1 Desember 2009 – 31 Desember 2010**.

| Atribut | Detail |
|---|---|
| **Sumber** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| **Link Data** | [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1jS45oopRhWJoP9TE4PK-LgoL3XvDXsDb/edit?gid=1081933993#gid=1081933993) |
| **Periode** | 01 Desember 2009 – 31 Desember 2010 |
| **Jenis Bisnis** | UK-based non-store online retail (giftware unik) |
| **Jumlah Instansi** | ±525.000 baris transaksi (Year 2009–2010) |
| **Lisensi** | CC BY 4.0 |

### Kolom Dataset

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `Invoice` | Nominal | Nomor invoice 6 digit unik per transaksi. Awalan `C` = pembatalan (cancellation). |
| `StockCode` | Nominal | Kode produk 5 digit unik per item. |
| `Description` | Nominal | Nama produk (item). Kolom ini digunakan sebagai **Item** dalam proses mining. |
| `Quantity` | Numerik | Jumlah unit produk yang dibeli per transaksi. |
| `InvoiceDate` | Numerik | Tanggal dan waktu transaksi. |
| `Price` | Numerik | Harga satuan produk dalam poundsterling (£). |
| `Customer ID` | Nominal | ID pelanggan unik 5 digit. |
| `Country` | Nominal | Negara domisili pelanggan. |

### Cara Memuat Dataset ke Aplikasi
1. Ekspor sheet data ke format **CSV** atau **Excel (.xlsx)**
2. Upload melalui sidebar aplikasi
3. Sesuaikan nama kolom: **Kolom Invoice** → `Invoice`, **Kolom Item** → `Description`

---

## 🚀 Fitur Aplikasi

Aplikasi terdiri dari **8 tab** yang saling terintegrasi:

### 📊 Tab 1 — Data Eksplorasi
Ringkasan statistik dataset: total transaksi, produk unik, rata-rata item per transaksi, distribusi item, top 15 produk terlaris, dan distribusi per negara.

### ⚙️ Tab 2 — Proses Mining
Penjelasan langkah-langkah algoritma Apriori secara interaktif beserta log proses saat mining berjalan (L1, L2, L3, waktu eksekusi).

### 📦 Tab 3 — Frequent Itemsets
Tabel seluruh frequent itemsets yang ditemukan, bisa difilter berdasarkan ukuran (1-itemset, 2-itemset, dst.), dilengkapi download CSV.

### 📜 Tab 4 — Association Rules
Tabel association rules dengan kolom support, confidence, lift, leverage, dan conviction. Ada highlight Top 5 rules terbaik berdasarkan lift.

### 📈 Tab 5 — Visualisasi
- Scatter plot Support vs Confidence (ukuran bubble = lift)
- Bar chart Top 10 Rules berdasarkan lift
- Bar chart Top 10 Item berdasarkan support
- Heatmap confidence antar item (Top 12)
- Histogram distribusi nilai lift

### 🛒 Tab 6 — Sistem Rekomendasi Produk
Pilih item yang sudah ada di keranjang → sistem merekomendasikan produk lain yang sering dibeli bersama berdasarkan confidence dan lift tertinggi. Dilengkapi visualisasi kekuatan rekomendasi.

### 🗂️ Tab 7 — SPK Tata Letak Rak (Rak A / Rak B)
Sistem Pendukung Keputusan untuk menentukan penempatan produk di rak toko:
- **Rak A** (strategis) → produk dengan skor tinggi, sering dibeli, ditempatkan di lorong utama
- **Rak B** (pendukung) → produk dengan frekuensi lebih rendah, ditempatkan di area pojok
- Rekomendasi penempatan berdampingan berdasarkan korelasi antar item

Skor dihitung dengan formula:
```
Skor = 0.5 × support + 0.3 × avg_confidence + 0.2 × avg_lift
```

### 🔮 Tab 8 — Prediksi Perilaku Pembelian
Input isi keranjang saat ini → prediksi item berikutnya yang paling mungkin dibeli, dengan estimasi probabilitas pembelian berbasis confidence rules:

| Kategori | Probabilitas |
|---|---|
| 🔴 Sangat Tinggi | ≥ 80% |
| 🟠 Tinggi | ≥ 60% |
| 🟡 Sedang | ≥ 40% |
| 🟢 Rendah | < 40% |

---

## 🛠️ Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```

### 4. Buka di Browser
Streamlit akan otomatis membuka `http://localhost:8501` di browser kamu.

---

## 📦 Dependencies

```
streamlit
pandas
numpy
plotly
openpyxl
```

Simpan sebagai `requirements.txt`:
```bash
pip freeze > requirements.txt
```

Atau install manual:
```bash
pip install streamlit pandas numpy plotly openpyxl
```

---

## ⚙️ Parameter Apriori

Bisa diatur melalui sidebar aplikasi:

| Parameter | Default | Keterangan |
|---|---|---|
| Minimum Support | 0.10 (10%) | Frekuensi minimum itemset muncul dalam semua transaksi |
| Minimum Confidence | 0.50 (50%) | Probabilitas minimum konsekuensi muncul saat anteseden ada |
| Minimum Lift | 1.00 | Lift > 1 berarti ada korelasi positif antar item |
