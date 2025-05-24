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

- Bagaimana cara memprediksi nilai review rating pelanggan berdasarkan data transaksi seperti jumlah pembelian, metode pembayaran, dan jenis produk yang dibeli?
- Faktor-faktor apa saja dalam transaksi penjualan yang memiliki pengaruh signifikan terhadap rating pelanggan?

### Goals

- Mengembangkan model machine learning yang mampu memprediksi nilai review rating pelanggan berdasarkan data transaksi.
- Mengidentifikasi variabel-variabel utama dalam data transaksi yang paling berpengaruh terhadap rating pelanggan.


### Solution statements

- Menggunakan beberapa algoritma regresi seperti KNN, Random Forest, dan AdaBoost untuk membandingkan performa model dalam memprediksi rating pelanggan.
- Membandingkan performa model dan memilih model terbaik berdasarkan metrik evaluasi seperti Mean Squared Error (MSE).

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
Berikut tahapan yang dilakukan dalam proses pra-pemrosesan data:

1. **Encoding Fitur Kategorikal**  
   - One-hot encoding pada kolom `Item Purchased` dan `Payment Method`
   - Konversi nilai boolean ke numerik (0/1)

2. **Pembersihan dan Ekstraksi Tanggal**  
   - Konversi `Date Purchase` ke datetime  
   - Ekstraksi `Purchase Month` dari tanggal  
   - Hapus kolom `Date Purchase`

3. **Handling Missing Values**  
   - Menghapus baris dengan nilai kosong pada `Purchase Amount (USD)` dan `Review Rating`

4. **Handling Outlier**  
   - Menggunakan metode IQR untuk menghapus outlier pada `Purchase Amount (USD)`

5. **Pemisahan Fitur dan Target**  
   - `X`: Fitur  
   - `y`: Target (`Review Rating`)

6. **Pembagian Data**  
   - Train: 80%  
   - Test: 20% (`random_state=42`)

7. **Standarisasi Data**  
   - Menggunakan `StandardScaler` untuk menormalkan fitur

8. **Reduksi Dimensi**  
   - PCA dengan `n_components=0.95` untuk mempertahankan 95% variansi

## Visualisasi

![image](https://github.com/user-attachments/assets/e08b5db6-64aa-4523-9531-692a44482347)


## Modeling
### Modeling Machine Learning Regression

### Deskripsi

Tahapan ini membahas model machine learning yang digunakan untuk menyelesaikan permasalahan regresi. Data sudah melalui reduksi dimensi dengan PCA sehingga pelatihan dilakukan pada fitur hasil PCA (`X_train_pca`).

---

### Algoritma yang Digunakan

1. **K-Nearest Neighbors Regressor (KNN)**  
   - Parameter utama: `n_neighbors=5`  
   - Prediksi dilakukan dengan rata-rata target dari 5 tetangga terdekat.

2. **Random Forest Regressor (RF)**  
   - Parameter utama: `n_estimators=100`, `random_state=42`  
   - Ensemble pohon keputusan, hasil prediksi rata-rata dari semua pohon.

3. **AdaBoost Regressor (Ada)**  
   - Parameter utama: `n_estimators=100`, `random_state=42`  
   - Metode boosting yang menggabungkan model lemah secara berturut-turut.

4. **Gradient Boosting Regressor (GB)**  
   - Parameter utama: `n_estimators=100`, `random_state=42`  
   - Boosting yang mengurangi error residual secara bertahap menggunakan gradien.

---

## Kelebihan dan Kekurangan Algoritma

| Algoritma      | Kelebihan                                              | Kekurangan                                             |
|----------------|--------------------------------------------------------|--------------------------------------------------------|
| **KNN**        | Sederhana, tidak perlu asumsi distribusi               | Sensitif skala, lambat untuk data besar, kurang efisien di dimensi tinggi |
| **Random Forest** | Robust terhadap overfitting, mudah tuning, kuat menangani data kompleks | Model besar, sulit interpretasi                        |
| **AdaBoost**   | Meningkatkan akurasi, adaptif terhadap kesalahan       | Sensitif noise dan outlier                              |
| **Gradient Boosting** | Biasanya akurasi terbaik, fleksibel loss function | Lambat pelatihan, mudah overfit tanpa tuning           |



## Pemilihan Model Terbaik

Setelah evaluasi dengan metrik seperti RMSE, MAE, dan R², model dengan performa terbaik dipilih sebagai solusi akhir. Misalnya:

- **Random Forest** dipilih karena kestabilan hasil dan akurasi yang konsisten lebih baik dibandingkan model lain.
- Pilihan ini juga didukung oleh kemudahan tuning dan kecepatan pelatihan dibanding boosting yang cenderung lebih sensitif overfitting.



## Rencana Improvement

- Melakukan **hyperparameter tuning** dengan Grid Search atau Random Search untuk mengoptimalkan parameter model terbaik (misal: `n_estimators`, `max_depth`, `learning_rate`).
- Mencoba teknik ensemble seperti stacking untuk meningkatkan performa model.



## Cara Menjalankan

1. Pastikan data sudah di-preprocessing dan direduksi dimensi menggunakan PCA.
2. Jalankan skrip pelatihan model untuk masing-masing algoritma.
3. Evaluasi hasil prediksi dan bandingkan metrik performa.
4. Pilih model terbaik berdasarkan evaluasi.


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

### Visualisasi dan Analisis MSE

![image](https://github.com/user-attachments/assets/e96bcfb8-6626-4ae4-8fa8-05a8e3789585)

### Kesimpulan

Dari hasil evaluasi MSE di atas, **model Boosting** dipilih sebagai model terbaik karena memberikan nilai error terkecil pada data test, yang merupakan indikasi kemampuan generalisasi terbaik dibandingkan model lainnya.

Untuk peningkatan performa, disarankan melakukan **hyperparameter tuning** pada model Boosting guna menurunkan nilai error lebih lanjut dan mengurangi potensi overfitting atau underfitting.

