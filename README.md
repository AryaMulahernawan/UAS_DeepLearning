# UAS_DeepLearning
Nama Anggota : 1. Arya Mulahernawan        (G1A022029)
               2. Muhammad Kevin Rinaldi   (G1A022059)
               3. Yebi Depriansyah         (G1A022063)

## Project Overview

Perkembangan teknologi Internet of Things (IoT) telah mendorong penggunaan sensor secara luas dalam berbagai sektor industri, termasuk sistem pemantauan dan pengelolaan infrastruktur air. Sensor-sensor ini secara kontinu menghasilkan data time series dalam jumlah besar yang merepresentasikan kondisi operasional mesin, seperti perubahan sinyal sensor dari waktu ke waktu. Meskipun data tersebut sangat berpotensi untuk digunakan dalam pemantauan kondisi sistem, kompleksitas dan volume data yang tinggi sering kali menyulitkan proses analisis secara manual, terutama dalam mendeteksi indikasi awal kegagalan mesin.

Dalam konteks sistem pompa air, kegagalan yang tidak terdeteksi sejak dini dapat menyebabkan gangguan serius bagi masyarakat, seperti terhentinya pasokan air dan meningkatnya biaya perbaikan akibat kerusakan yang bersifat reaktif. Oleh karena itu, dibutuhkan sebuah sistem yang mampu mendeteksi anomali pada data sensor secara otomatis sebagai upaya pencegahan terhadap kegagalan sistem. Deteksi anomali yang akurat memungkinkan penerapan predictive maintenance, sehingga potensi kerusakan dapat diidentifikasi sebelum berkembang menjadi kegagalan total.

Untuk menjawab tantangan tersebut, proyek ini dirancang untuk mengembangkan sebuah sistem deteksi anomali pada data sensor IoT menggunakan pendekatan deep learning. Sistem yang dikembangkan memanfaatkan data historis sensor multivariat yang dikumpulkan dari sistem pompa air, dengan tujuan mengklasifikasikan kondisi operasional mesin ke dalam dua kategori utama, yaitu kondisi normal dan kondisi anomali. Data time series diproses menggunakan pendekatan sliding window agar pola temporal dapat ditangkap secara lebih efektif, sehingga model mampu mengenali perubahan perilaku sensor yang mengindikasikan kondisi tidak normal.

Urgensi dari proyek ini terlihat dari beberapa aspek. Pertama, sistem deteksi anomali otomatis dapat mengurangi ketergantungan pada pemantauan manual dan mempercepat proses identifikasi gangguan pada sistem pompa. Kedua, penerapan sistem ini berpotensi menurunkan biaya operasional melalui perawatan berbasis kondisi (condition-based maintenance), bukan perawatan reaktif. Ketiga, sistem ini dapat meningkatkan keandalan dan kontinuitas layanan pompa air, yang secara langsung berdampak pada kualitas hidup masyarakat yang bergantung pada infrastruktur tersebut.

Dari perspektif ilmiah, deteksi anomali pada data time series sensor merupakan topik yang banyak diteliti dalam bidang machine learning dan deep learning. Chandola et al. (2009) menyatakan bahwa deteksi anomali bertujuan untuk mengidentifikasi pola data yang menyimpang secara signifikan dari perilaku normal dan sering kali berkaitan dengan kejadian kritis dalam suatu sistem. Selain itu, Malhotra et al. (2015) menunjukkan bahwa model berbasis Long Short-Term Memory (LSTM) efektif dalam memodelkan dependensi temporal jangka panjang pada data sensor. Penggabungan Convolutional Neural Network (CNN) dan LSTM dalam arsitektur hibrida semakin banyak digunakan karena kemampuannya dalam mengekstraksi fitur lokal sekaligus memodelkan dinamika waktu secara bersamaan, sehingga sangat relevan untuk kasus deteksi anomali pada sistem IoT.

Referensi

Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. ACM Computing Surveys, 41(3), 1–58. https://doi.org/10.1145/1541880.1541882

Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2015). Long short term memory networks for anomaly detection in time series. Proceedings of the European Symposium on Artificial Neural Networks (ESANN).https://www.researchgate.net/publication/304782562_Long_Short_Term_Memory_Networks_for_Anomaly_Detection_in_Time_Series

## Business Understanding

Sistem pemantauan berbasis sensor IoT memegang peranan penting dalam pengelolaan dan pemeliharaan infrastruktur industri, termasuk sistem pompa air. Sensor-sensor yang terpasang pada mesin pompa secara kontinu menghasilkan data time series yang merepresentasikan kondisi operasional sistem. Namun, tingginya volume dan kompleksitas data sensor tersebut membuat proses pemantauan secara manual menjadi sulit dan tidak efisien, terutama dalam mendeteksi indikasi awal terjadinya kegagalan sistem.

Pada sistem pompa air, kegagalan mesin dapat menyebabkan gangguan serius terhadap pasokan air dan berdampak langsung pada aktivitas serta kualitas hidup masyarakat. Selain itu, keterlambatan dalam mendeteksi kerusakan dapat meningkatkan biaya perawatan akibat tindakan perbaikan yang bersifat reaktif. Oleh karena itu, dibutuhkan sebuah sistem deteksi anomali yang cerdas dan andal untuk membantu mengidentifikasi kondisi tidak normal pada mesin pompa secara otomatis dan tepat waktu.

Proyek ini bertujuan untuk membangun sistem deteksi anomali berbasis data sensor IoT dengan memanfaatkan data historis sensor yang tersedia. Sistem yang dikembangkan tidak bergantung pada interpretasi fisik spesifik dari masing-masing sensor, melainkan berfokus pada pola temporal dan hubungan antar sensor dalam membedakan kondisi normal dan kondisi anomali. Dengan pendekatan ini, sistem diharapkan mampu mendukung penerapan predictive maintenance pada sistem pompa air.

### Problem Statements

Berdasarkan latar belakang permasalahan tersebut, rumusan masalah dalam proyek ini adalah sebagai berikut:

1. Bagaimana cara membangun sistem deteksi anomali pada data sensor IoT berbasis time series yang mampu membedakan kondisi normal dan kondisi anomali pada sistem pompa air?

2. Bagaimana cara memanfaatkan arsitektur hybrid CNN–LSTM untuk mengekstraksi fitur spasial dan temporal dari data sensor multivariat?

3. Bagaimana cara mengukur dan mengevaluasi kinerja model deteksi anomali secara kuantitatif pada data yang memiliki ketidakseimbangan kelas?

### Goals

Tujuan yang ingin dicapai dalam proyek ini adalah sebagai berikut:

1. Mengembangkan model deteksi anomali berbasis deep learning yang mampu mengklasifikasikan kondisi sistem pompa air ke dalam kategori normal dan anomali berdasarkan data sensor IoT.

2. Menerapkan pendekatan sliding window untuk menangkap pola temporal pada data time series sensor.

3. Mengevaluasi kinerja model deteksi anomali menggunakan metrik evaluasi yang relevan, seperti accuracy, precision, recall, F1-score, dan ROC-AUC.

4. Membandingkan performa model CNN–LSTM dengan model baseline seperti Support Vector Machine (SVM) dan Random Forest.

### Solution Statements
Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, proyek ini menggunakan beberapa pendekatan pemodelan sebagai berikut:

1. Hybrid CNN–LSTM (Model Utama)

Pendekatan utama yang digunakan adalah arsitektur hybrid Convolutional Neural Network (CNN) dan Long Short-Term Memory (LSTM). CNN digunakan untuk mengekstraksi fitur lokal dari setiap window data sensor, sedangkan LSTM digunakan untuk memodelkan ketergantungan temporal jangka panjang dalam data time series. Kombinasi kedua arsitektur ini diharapkan mampu menangkap pola kompleks yang mengindikasikan kondisi anomali pada sistem pompa air.

Data sensor multivariat diproses menggunakan pendekatan sliding window, kemudian dinormalisasi sebelum dimasukkan ke dalam model. Model dilatih menggunakan teknik regulasi seperti dropout, early stopping, dan class weighting untuk mengatasi permasalahan overfitting dan ketidakseimbangan kelas.

2. Model Baseline (SVM dan Random Forest)

Sebagai pembanding terhadap model deep learning, digunakan model baseline berupa Support Vector Machine (SVM) dan Random Forest. Pada pendekatan ini, data time series diubah menjadi representasi vektor satu dimensi melalui proses flattening pada setiap window. Hasil evaluasi dari model baseline digunakan untuk menilai sejauh mana peningkatan kinerja yang diperoleh dari penggunaan arsitektur CNN–LSTM.


## Data Understanding

Pada tahap Data Understanding, dilakukan pemahaman awal terhadap dataset yang digunakan dalam proyek pengembangan sistem deteksi anomali pada data sensor IoT menggunakan arsitektur hybrid CNN–LSTM. Tahapan ini bertujuan untuk memahami karakteristik data, struktur dataset, kondisi data, serta informasi yang terkandung dalam setiap fitur sebelum dilakukan proses prapemrosesan dan pemodelan.

### Sumber Dataset

Dataset yang digunakan dalam penelitian ini diperoleh dari platform Kaggle dengan judul Pump Sensor Data, yang dapat diakses melalui tautan berikut:
https://www.kaggle.com/datasets/nphantawee/pump-sensor-data

Dataset ini berasal dari sistem pemantauan pompa air di sebuah wilayah terpencil. Latar belakang pengumpulan data ini didasarkan pada permasalahan nyata, di mana dalam satu tahun terakhir terjadi beberapa kegagalan sistem pompa yang menyebabkan gangguan serius terhadap pasokan air dan kehidupan masyarakat sekitar. Meskipun data sensor tersedia, tim pemeliharaan mengalami kesulitan dalam mengidentifikasi pola yang jelas sebelum terjadinya kegagalan sistem. Oleh karena itu, dataset ini diharapkan dapat dimanfaatkan untuk memprediksi atau mendeteksi potensi kegagalan sistem lebih awal melalui pendekatan berbasis data.

### Deskripsi Umum Dataset

Dataset ini merupakan data time series multivariat yang terdiri dari pembacaan sensor IoT yang direkam secara berkala. Berdasarkan hasil eksplorasi awal, karakteristik dataset adalah sebagai berikut:

- Jumlah baris data: 220.320 baris (sebelum pembersihan data)

- Jumlah kolom: 55 kolom

- Ukuran memori: sekitar 92,5 MB

Tipe data:

- 52 kolom bertipe numerik (float64)

- 1 kolom bertipe integer (int64)

- 2 kolom bertipe objek (object)

Setiap baris data merepresentasikan kondisi sistem pompa air pada satu titik waktu tertentu.

### Struktur dan Penjelasan Fitur

Dataset terdiri dari kolom-kolom berikut:

1. Unnamed: 0
Merupakan indeks bawaan hasil proses penyimpanan data, yang tidak memiliki makna informatif terhadap kondisi sistem dan tidak digunakan dalam proses analisis.

2. timestamp
Menunjukkan waktu pencatatan data sensor dalam format waktu (datetime). Kolom ini digunakan untuk menjaga urutan temporal data.

3. sensor_00 hingga sensor_51
Merupakan data pembacaan dari 52 sensor IoT yang terpasang pada sistem pompa air. Seluruh sensor menyajikan nilai numerik mentah (raw values) tanpa informasi eksplisit mengenai jenis atau parameter fisik yang diukur. Oleh karena itu, masing-masing sensor diperlakukan sebagai variabel independen yang secara kolektif merepresentasikan kondisi operasional sistem pompa.

4. machine_status
Menunjukkan status operasional sistem pompa pada waktu tertentu dengan tiga kategori utama, yaitu:

   - NORMAL: sistem beroperasi dalam kondisi normal,

   - RECOVERING: sistem berada dalam fase pemulihan,

   - BROKEN: sistem mengalami kegagalan atau kerusakan.

Kolom machine_status digunakan sebagai label dalam proses deteksi anomali.

### Kondisi dan Kualitas Data

Berdasarkan hasil pemeriksaan awal terhadap dataset, diperoleh beberapa temuan terkait kondisi data sebagai berikut:

1. Terdapat missing values pada sejumlah kolom sensor, dengan tingkat yang bervariasi.

2. Kolom sensor_15 memiliki nilai kosong sepenuhnya (0 non-null), sehingga kolom ini tidak dapat digunakan dalam analisis lebih lanjut.

3. Beberapa sensor lain, seperti sensor_00, sensor_06, sensor_07, sensor_50, dan sensor_51, juga memiliki jumlah data yang hilang dalam jumlah signifikan.

4. Tidak ditemukan data duplikat pada dataset.

5. Data sensor berpotensi mengandung outlier, yang dapat mencerminkan kondisi tidak normal atau gangguan pada sistem pompa.

Permasalahan missing values dan outlier ini menjadi dasar dilakukannya proses pembersihan dan prapemrosesan data sebelum tahap pemodelan.

### Transformasi Label

Dalam penelitian ini, permasalahan difokuskan pada deteksi anomali secara biner. Oleh karena itu, kolom machine_status ditransformasikan menjadi dua kelas utama sebagai berikut:

1. NORMAL → 0 (Normal)

2. RECOVERING dan BROKEN → 1 (Anomali)

Pendekatan ini bertujuan untuk menyederhanakan permasalahan menjadi klasifikasi dua kelas, dengan fokus utama pada pendeteksian kondisi tidak normal sebagai indikasi awal potensi kegagalan sistem pompa air.

### Alasan Pemilihan Dataset

Dataset Pump Sensor Data dipilih dalam penelitian ini berdasarkan beberapa pertimbangan berikut:

1. Dataset berasal dari kasus nyata sistem IoT industri, sehingga hasil penelitian memiliki relevansi praktis.

2. Data berbentuk multivariate time series, yang sesuai untuk penerapan model deep learning berbasis CNN dan LSTM.

3. Tersedianya label kondisi mesin memungkinkan penerapan pendekatan supervised learning untuk deteksi anomali.

4. Dataset mendukung pengembangan sistem predictive maintenance guna meningkatkan keandalan dan efisiensi operasional sistem pompa air.


| Judul Kolom 1 | Judul Kolom 2 |
| :--- | :--- |
| Tulis konten Anda di sini. | Tulis konten kedua di sini. |
| Baris baru kiri. | Baris baru kanan. |

