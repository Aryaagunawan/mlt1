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

- Menggunakan beberapa algoritma regresi seperti Linear Regression, Decision Tree Regressor, dan Random Forest Regressor untuk membandingkan performa model dalam memprediksi rating pelanggan.
- Melakukan hyperparameter tuning terhadap model terbaik untuk meningkatkan akurasi prediksi berdasarkan metrik evaluasi seperti MAE (Mean Absolute Error) dan RMSE (Root Mean Squared Error).
- Menggunakan teknik feature importance untuk mengevaluasi variabel yang paling berpengaruh terhadap prediksi rating pelanggan.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

