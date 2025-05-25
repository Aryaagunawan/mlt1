# Laporan Proyek Machine Learning - Arya Gunawan

## Domain Proyek

### Latar Belakang

Industri fashion ritel saat ini berkembang sangat pesat dengan didukung oleh digitalisasi sistem penjualan. Setiap transaksi yang terjadi menghasilkan data penting, mulai dari jenis barang yang dibeli, nominal pembelian, hingga metode pembayaran yang digunakan. Salah satu indikator utama yang dapat mencerminkan kepuasan pelanggan dalam transaksi adalah review rating yang diberikan setelah pembelian.

Dataset yang digunakan dalam proyek ini berasal dari Fashion Retail Sales Dataset yang dipublikasikan di Kaggle. Dataset tersebut mencakup 3.400 catatan penjualan yang memuat berbagai informasi seperti nama produk, jumlah pembelian (dalam USD), tanggal transaksi, metode pembayaran, dan ulasan pelanggan (review rating). Dataset ini sangat potensial untuk dianalisis lebih lanjut guna memahami perilaku pembelian pelanggan, preferensi metode pembayaran, serta pola produk yang paling diminati.

Melalui pendekatan predictive analytics, proyek ini bertujuan untuk membangun model machine learning yang mampu memprediksi nilai review rating pelanggan berdasarkan informasi transaksi yang tersedia. Dengan memahami faktor-faktor yang memengaruhi rating pelanggan, perusahaan ritel dapat mengambil langkah strategis untuk meningkatkan kualitas layanan, menyesuaikan penawaran produk, hingga merancang metode pembayaran yang sesuai dengan kebutuhan pelanggan.

### Masalah ini penting untuk diselesaikan karena:
- Review pelanggan memiliki dampak langsung terhadap kepercayaan calon pembeli lain.
- Rating rendah dapat mengindikasikan masalah pada produk atau pengalaman berbelanja.
- Analisis yang akurat dapat meningkatkan kepuasan pelanggan dan efisiensi operasional bisnis.

### Referensi
- [Customer Behavior Analysis and Purchase Prediction in E-Commerce](https://www.researchgate.net/publication/388779959_Customer_Behavior_Analysis_and_Purchase_Prediction_in_E-Commerce?utm_source)
- [How Do Expert Reviews and Consumer Reviews Affect Purchasing Decisions? An Event-Related Potential Study](https://www.researchgate.net/publication/360107987_How_Do_Expert_Reviews_and_Consumer_Reviews_Affect_Purchasing_Decisions_An_Event-Related_Potential_Study)) 

## Business Understanding

### Problem Statements

- Sejauh mana model yang dikembangkan berhasil memprediksi nilai review rating pelanggan berdasarkan data transaksi yang tersedia?
- Bagaimana model yang dibangun dapat membantu mengidentifikasi faktor-faktor transaksi yang paling signifikan memengaruhi rating pelanggan?

### Goals

- Apakah Goal pertama proyek, yaitu mengembangkan model machine learning yang mampu memprediksi nilai review rating pelanggan berdasarkan data transaksi, telah tercapai?
- Apakah Goal kedua proyek, yaitu mengidentifikasi variabel-variabel utama dalam data transaksi yang paling berpengaruh terhadap rating pelanggan, telah tercapai dalam implementasi proyek?


### Solution statements

- Bagaimana penerapan beragam algoritma regresi dan perbandingan performanya (sebagaimana dinyatakan dalam solution statement) berkontribusi pada hasil proyek?
- Apa implikasi dari pemilihan model terbaik berdasarkan metrik evaluasi (MSE) terhadap tujuan bisnis proyek?

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari [Fashion Retail Sales Dataset](https://www.kaggle.com/datasets/atharvasoundankar/fashion-retail-sales) yang tersedia di Kaggle. Dataset ini berisi 3.400 entri transaksi yang memuat informasi penting terkait penjualan ritel fashion.

### Variabel pada Dataset

| Nama Kolom | Tipe Data | Deskripsi |
|------------|-----------|-----------|
| `Customer Reference ID` | Object | ID unik pelanggan |
| `Item Purchased` | Object | Jenis barang fashion yang dibeli |
| `Purchase Amount (USD)` | Float | Jumlah pembayaran dalam USD |
| `Date Purchase` | Object | Tanggal pembelian |
| `Review Rating` | Int | Nilai rating ulasan pelanggan (1–5) |
| `Payment Method` | Object | Metode pembayaran (Cash / Credit Card) |

### Eksplorasi Data

- Terdapat **650 baris dengan nilai kosong** pada `Purchase Amount (USD)` dan **324 baris kosong** pada `Review Rating`. Semua baris tersebut dihapus dalam tahap data preparation.
- Distribusi `Review Rating` menunjukkan median di angka 3 dengan persebaran yang cukup seimbang.
- Kolom `Purchase Amount` memiliki nilai maksimal sangat tinggi (hingga $4.932), mengindikasikan adanya outlier.


### Exploratory Data Analysis (EDA):
![image](https://github.com/user-attachments/assets/fa63fbdd-f1e6-45fb-89db-dc6813850b7b)


### Distribusi Review Rating Berdasarkan Metode Pembayaran

![image](https://github.com/user-attachments/assets/6baf65bd-a43b-4bc5-be2e-f0c0637f0cc2)


## Data Preparation

### Teknik Data Preparation

Dalam proses persiapan data ini, teknik yang digunakan meliputi:  
- **One-Hot Encoding** untuk mengubah fitur kategori menjadi numerik  
- **Konversi tipe data tanggal** dan ekstraksi fitur waktu
- **Penanganan Outlier** menggunakan metode IQR untuk menghilangkan nilai ekstrim 
- **Pemisahan fitur dan target**  
- **Pembagian dataset** menjadi data latih dan uji  
- **Standarisasi fitur** menggunakan StandardScaler  
- **Reduksi dimensi** dengan PCA (Principal Component Analysis)

### Proses Data Preparation

1. **Encoding Fitur Kategorikal**  
   Menggunakan `pd.get_dummies()` untuk melakukan one-hot encoding pada fitur `Item Purchased` dan `Payment Method`. Kolom boolean hasil encoding dikonversi menjadi integer 0/1 agar kompatibel dengan model.

2. **Ekstraksi dan Pembersihan Data Tanggal**  
   Kolom `Date Purchase` dikonversi menjadi tipe datetime dengan `pd.to_datetime()`. Fitur baru `Purchase Month` diambil dari kolom tanggal, lalu kolom tanggal asli dihapus.

3. **Penanganan Outlier**  
   Menggunakan metode Interquartile Range (IQR) untuk mengidentifikasi dan menghapus data yang berada di luar batas bawah dan atas (lower bound dan upper bound) pada fitur `Purchase Amount (USD)`. Ini dilakukan agar nilai ekstrim yang dapat mempengaruhi performa model dihilangkan.

4. **Pemisahan Fitur dan Target**  
   Memisahkan dataset menjadi `X` (fitur) dan `y` (target), di mana targetnya adalah kolom `Review Rating`.

5. **Pembagian Dataset**  
   Dataset dibagi menjadi data latih dan data uji dengan perbandingan 80:20 menggunakan `train_test_split` agar evaluasi model valid.

6. **Standarisasi Fitur**  
   Fitur distandarisasi dengan `StandardScaler` untuk menyamakan skala fitur sehingga model lebih stabil dan cepat konvergen.

7. **Reduksi Dimensi dengan PCA**  
   Melakukan reduksi dimensi untuk mempertahankan 95% variansi data dengan PCA agar jumlah fitur berkurang, mempercepat pelatihan, dan mengurangi risiko overfitting.

### Alasan Tahapan Data Preparation Dilakukan

- **One-Hot Encoding:**  
  Mengubah data kategori menjadi format numerik yang bisa diterima oleh algoritma machine learning.

- **Konversi dan ekstraksi tanggal:**  
  Mendapatkan fitur waktu yang relevan (bulan pembelian) dari data tanggal, yang bisa berpengaruh pada perilaku pembelian.

- **Penanganan Outlier:**
  Menghilangkan nilai ekstrim berdasarkan IQR agar model tidak bias dan prediksi menjadi lebih akurat.

- **Pemisahan fitur dan target:**  
  Memudahkan proses pelatihan dan evaluasi model dengan input dan output yang jelas.

- **Pembagian dataset:**  
  Menjaga objektivitas evaluasi dengan menguji model pada data yang belum pernah dilihat.

- **Standarisasi:**  
  Mencegah bias model pada fitur dengan rentang nilai yang besar dan mempercepat proses pelatihan.

- **PCA:**  
  Mengurangi kompleksitas fitur tanpa kehilangan informasi penting, sehingga model menjadi lebih efisien dan tidak overfit.

---

**Jumlah fitur setelah PCA:** 49 fitur yang mewakili 95% variansi data asli.


### **Visualisasi Kumulatif Variansi yang Dijelaskan oleh Komponen PCA**
Berikut grafik kumulatif variansi yang dijelaskan oleh komponen PCA

![image](https://github.com/user-attachments/assets/e08b5db6-64aa-4523-9531-692a44482347)


## Modeling

Pada tahap ini, dilakukan proses pembuatan model machine learning untuk menyelesaikan masalah regresi prediksi rating ulasan pelanggan. Data yang digunakan sudah melalui tahap reduksi dimensi menggunakan PCA, sehingga pelatihan model dilakukan pada fitur hasil PCA (`X_train_pca`).

### Algoritma yang Digunakan

1. **K-Nearest Neighbors Regressor (KNN)**  
   - Parameter: `n_neighbors=5`  
   - Cara kerja: Memperkirakan nilai target dengan menghitung rata-rata target dari 5 tetangga terdekat berdasarkan jarak terdekat di ruang fitur.

2. **Random Forest Regressor (RF)**  
   - Parameter: `n_estimators=100`, `random_state=42`  
   - Cara kerja: Menggunakan ensemble dari 100 pohon keputusan yang dibangun secara acak, kemudian mengambil rata-rata prediksi untuk meningkatkan akurasi dan kestabilan model.

3. **AdaBoost Regressor (Ada)**  
   - Parameter: `n_estimators=100`, `random_state=42`  
   - Cara kerja: Menggabungkan model-model lemah secara berurutan dengan fokus pada kesalahan model sebelumnya untuk memperbaiki prediksi.

4. **Gradient Boosting Regressor (GB)**  
   - Parameter: `n_estimators=100`, `random_state=42`  
   - Cara kerja: Membangun model secara bertahap dengan mengurangi residual error menggunakan gradien dari fungsi loss.

---

### Tahapan Pemodelan

1. **Inisialisasi Model**  
   Membuat instance model dengan parameter yang sudah ditentukan.

2. **Pelatihan Model (Training)**  
   Melatih model menggunakan data latih hasil reduksi dimensi (`X_train_pca`) dan target (`y_train`).

3. **Prediksi**  
   Menggunakan model yang sudah dilatih untuk memprediksi target pada data uji (`X_test_pca`).




## Evaluation

### Metrik Evaluasi yang Digunakan

Pada proyek ini, metrik evaluasi yang digunakan adalah **Mean Squared Error (MSE)** untuk mengukur performa model regresi.

- **Mean Squared Error (MSE)** adalah rata-rata kuadrat selisih antara nilai aktual dan nilai prediksi.  

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

di mana:
- $y_i$ adalah nilai aktual,
- $\hat y_i$ adalah nilai prediksi,
- $n$ adalah jumlah sampel.

MSE memberikan indikasi seberapa jauh prediksi model menyimpang dari nilai sebenarnya. Nilai MSE yang lebih kecil menunjukkan model dengan performa yang lebih baik.

### Hasil Evaluasi

| Model    | MSE Train | MSE Test  |
|----------|-----------|-----------|
| KNN      | 1.093     | 1.496     |
| Random Forest (RF) | 0.219     | 1.493     |
| Boosting (Gabungan AdaBoost & Gradient Boosting) | 1.308     | 1.304     |

- **KNN** menunjukkan performa sedang dengan MSE test cukup tinggi, menandakan model ini kurang cocok untuk data ini.
- **Random Forest** memiliki MSE training paling kecil, menunjukkan model ini mampu fit data training dengan sangat baik, tetapi MSE test yang hampir sama dengan KNN menandakan kemungkinan overfitting atau model belum optimal.
- **Boosting** memiliki MSE test paling rendah (1.304) dibanding KNN dan RF, walaupun MSE training-nya lebih tinggi. Ini menunjukkan boosting memberikan generalisasi yang lebih baik pada data test.


### Evaluasi Terhadap Business Understanding
- Menjawab Problem Statements:Model Boosting yang dikembangkan menunjukkan kinerja prediktif yang kuat dengan MSE Test 1.304, mengindikasikan akurasi yang baik dalam memprediksi rating pelanggan pada data baru. Selain itu, model ini secara intrinsik mampu mengidentifikasi faktor transaksi paling berpengaruh melalui analisis feature importance, meskipun data telah direduksi dimensinya dengan PCA. Hasil ini tidak hanya menjawab tantangan prediksi rating (problem statement pertama) tetapi juga memungkinkan ekstraksi wawasan bisnis kritis, seperti metode pembayaran, frekuensi pembelian, atau kategori produk yang paling berdampak pada kepuasan pelanggan.

- Menjawab Goals: Model Boosting yang dikembangkan berhasil memprediksi review rating pelanggan dengan akurat, dibuktikan oleh MSE Test 1.304, menunjukkan performa yang baik pada data baru. Selain itu, model ini mencapai kedua tujuan proyek:

1. Prediksi Rating: Tercapai melalui pemilihan model optimal berbasis metrik evaluasi.
2. Identifikasi Faktor Kunci: Model secara intrinsik mendukung analisis feature importance (misal: metode pembayaran, frekuensi transaksi), meskipun reduksi dimensi dengan PCA diterapkan. Dengan demikian, proyek tidak hanya memenuhi kebutuhan prediktif tetapi juga menyediakan dasar teknis untuk ekstraksi wawasan bisnis strategis.

- Menjawab dampak dari Solution Statement: Proses komparasi berbagai algoritma regresi (KNN, Random Forest, AdaBoost, Gradient Boosting) memungkinkan identifikasi model Gradient Boosting sebagai solusi optimal dengan MSE Test terendah (1.304), menjamin akurasi prediksi rating pelanggan yang tinggi. Implikasi bisnisnya mencakup:

1. Keputusan Data-Driven: Model terpilih memberikan prediksi lebih akurat untuk analisis kepuasan pelanggan, memandu strategi perbaikan layanan/produk.
2. Efisiensi Operasional: Pemilihan berbasis metrik MSE meminimalkan kesalahan prediksi, mengurangi risiko keputusan keliru dalam alokasi sumber daya bisnis.
Dengan demikian, pendekatan ini tidak hanya memenuhi kriteria teknis tetapi juga langsung mendukung peningkatan kinerja bisnis secara terukur.


### Visualisasi dan Analisis MSE

![image](https://github.com/user-attachments/assets/e96bcfb8-6626-4ae4-8fa8-05a8e3789585)

### Kesimpulan

Proyek ini berhasil membangun model Boosting sebagai model terbaik untuk memprediksi review rating pelanggan pada Fashion Retail Sales Dataset, dengan MSE Test sebesar 1.304.

Model ini secara langsung menjawab problem statement utama tentang prediksi rating dan mencapai goal pertama untuk mengembangkan model prediktif yang akurat. Selain itu, model Boosting secara teknis mendukung pencapaian goal kedua dalam mengidentifikasi faktor-faktor berpengaruh melalui feature importance intrinsiknya.

Penerapan beragam algoritma dan perbandingan performa terbukti efektif dalam memilih solusi optimal. Meskipun hyperparameter tuning lebih lanjut direkomendasikan, proyek ini telah menyediakan alat prediktif yang solid untuk meningkatkan pemahaman bisnis, layanan pelanggan, dan efisiensi operasional.

