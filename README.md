# UAS_DeepLearning
Nama Anggota : 
               1. Arya Mulahernawan        (G1A022029)

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


| No | Column | Non-Null Count | Dtype |
| :--- | :--- | :--- | :--- |
| 0 | Unnamed: 0 | 220320 non-null | int64 |
| 1 | timestamp | 220320 non-null | object |
| 2 | sensor_00 | 210112 non-null | float64 |
| 3 | sensor_01 | 219951 non-null | float64 |
| 4 | sensor_02 | 220301 non-null | float64 |
| 5 | sensor_03 | 220301 non-null | float64 |
| 6 | sensor_04 | 220301 non-null | float64 |
| 7 | sensor_05 | 220301 non-null | float64 |
| 8 | sensor_06 | 215522 non-null | float64 |
| 9 | sensor_07 | 214869 non-null | float64 |
| 10 | sensor_08 | 215213 non-null | float64 |
| 11 | sensor_09 | 215725 non-null | float64 |
| 12 | sensor_10 | 220301 non-null | float64 |
| 13 | sensor_11 | 220301 non-null | float64 |
| 14 | sensor_12 | 220301 non-null | float64 |
| 15 | sensor_13 | 220301 non-null | float64 |
| 16 | sensor_14 | 220299 non-null | float64 |
| 17 | sensor_15 | 0 non-null | float64 |
| 18 | sensor_16 | 220289 non-null | float64 |
| 19 | sensor_17 | 220274 non-null | float64 |
| 20 | sensor_18 | 220274 non-null | float64 |
| 21 | sensor_19 | 220304 non-null | float64 |
| 22 | sensor_20 | 220304 non-null | float64 |
| 23 | sensor_21 | 220304 non-null | float64 |
| 24 | sensor_22 | 220279 non-null | float64 |
| 25 | sensor_23 | 220304 non-null | float64 |
| 26 | sensor_24 | 220304 non-null | float64 |
| 27 | sensor_25 | 220284 non-null | float64 |
| 28 | sensor_26 | 220300 non-null | float64 |
| 29 | sensor_27 | 220304 non-null | float64 |
| 30 | sensor_28 | 220304 non-null | float64 |
| 31 | sensor_29 | 220248 non-null | float64 |
| 32 | sensor_30 | 220059 non-null | float64 |
| 33 | sensor_31 | 220304 non-null | float64 |
| 34 | sensor_32 | 220252 non-null | float64 |
| 35 | sensor_33 | 220304 non-null | float64 |
| 36 | sensor_34 | 220304 non-null | float64 |
| 37 | sensor_35 | 220304 non-null | float64 |
| 38 | sensor_36 | 220304 non-null | float64 |
| 39 | sensor_37 | 220304 non-null | float64 |
| 40 | sensor_38 | 220293 non-null | float64 |
| 41 | sensor_39 | 220293 non-null | float64 |
| 42 | sensor_40 | 220293 non-null | float64 |
| 43 | sensor_41 | 220293 non-null | float64 |
| 44 | sensor_42 | 220293 non-null | float64 |
| 45 | sensor_43 | 220293 non-null | float64 |
| 46 | sensor_44 | 220293 non-null | float64 |
| 47 | sensor_45 | 220293 non-null | float64 |
| 48 | sensor_46 | 220293 non-null | float64 |
| 49 | sensor_47 | 220293 non-null | float64 |
| 50 | sensor_48 | 220293 non-null | float64 |
| 51 | sensor_49 | 220293 non-null | float64 |
| 52 | sensor_50 | 143303 non-null | float64 |
| 53 | sensor_51 | 204937 non-null | float64 |
| 54 | machine_status | 220320 non-null | object |




| Metric | Unnamed: 0 | timestamp | sensor_00 | ... | sensor_50 | sensor_51 | label |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **count** | 220320.0 | 220320 | 210112.0 | ... | 143303.0 | 204937.0 | 220320.0 |
| **mean** | 110159.5 | 2018-06-16* | 4882.37 | ... | 183.04 | 202.69 | 0.065 |
| **min** | 0.0 | 2018-04-01 | 0.0 | ... | 27.48 | 27.77 | 0.0 |
| **25%** | 55079.75 | 2018-05-09 | 2.43 | ... | 167.53 | 179.10 | 0.0 |
| **50%** | 110159.5 | 2018-06-16 | 2.45 | ... | 193.86 | 197.33 | 0.0 |
| **75%** | 165239.25 | 2018-07-24 | 2.49 | ... | 219.90 | 216.72 | 0.0 |
| **max** | 220319.0 | 2018-08-31 | 2.54 | ... | 1000.0 | 1000.0 | 1.0 |
| **std** | 63601.04 | NaN | 0.41 | ... | 65.25 | 109.58 | 0.24 |

> *Catatan: Kolom tengah disingkat (...) untuk keterbacaan.*
</details>


<img width="597" height="455" alt="image" src="https://github.com/user-attachments/assets/8d96455b-bc38-40d2-a41d-0eb54c8f96e1" />

Pada gambar ditampilkan distribusi kelas label:

- Kelas Normal (0) jauh lebih dominan dibandingkan kelas Anomali (1).

- Dataset bersifat imbalanced, dengan jumlah data anomali yang jauh lebih sedikit.

<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/74becf7b-8fe3-4926-9277-912dd6b0d6dc" />
<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/14016af8-8e04-42bb-a36c-dc2cf6e3cf22" />
<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/5bf16c3b-ebc8-48d5-8da4-467cb36f91b2" />



Pada beberapa gambar diatas adalah sampel dari data sensor yaitu sansor_00, sensor_01 dan sensor_51, ditampilkan visualisasi data time series dari beberapa sensor terhadap waktu (timestamp), dengan titik berwarna merah yang menandakan kondisi anomali.

Interpretasi:

- Garis biru menunjukkan nilai sensor yang berubah secara kontinu dari waktu ke waktu.

- Titik merah merepresentasikan data yang berlabel anomali (RECOVERING atau BROKEN).

- Terlihat bahwa pada periode tertentu terjadi penurunan atau lonjakan nilai sensor yang signifikan, yang berasosiasi dengan kondisi anomali.

- Pola anomali tidak selalu berupa satu titik tunggal, melainkan sering muncul dalam rentang waktu tertentu, menunjukkan bahwa gangguan sistem bersifat temporal.
  
Dari beberapa plot sensor yang berbeda, terlihat bahwa:

- Setiap sensor memiliki rentang nilai (skala) yang sangat berbeda, misalnya ada sensor dengan nilai di kisaran kecil, sementara sensor lain memiliki nilai hingga ratusan atau ribuan.

- Beberapa sensor menunjukkan fluktuasi relatif stabil saat kondisi normal, namun berubah drastis saat anomali terjadi.



## Data Preparation

Tahap Data Preparation dilakukan untuk memastikan data sensor IoT yang digunakan memiliki kualitas yang baik dan siap digunakan dalam proses pemodelan deteksi anomali menggunakan arsitektur hybrid CNN–LSTM. Tahapan ini mencakup pemilihan fitur, penanganan nilai hilang, penghapusan data duplikat, normalisasi data, serta pembentukan data time series dalam bentuk sliding window.

### 1. Label Transformation

**Proses:**
Kolom machine_status ditransformasikan menjadi label biner sebagai berikut:

NORMAL → 0 (Normal)

RECOVERING dan BROKEN → 1 (Anomali)

Alasan:
Transformasi ini dilakukan untuk menyederhanakan permasalahan menjadi deteksi anomali biner, dengan fokus pada identifikasi kondisi tidak normal sebagai indikasi awal kegagalan sistem.

### 2. Missing Value Handling

**Proses:**
Penanganan nilai hilang dilakukan melalui beberapa langkah, yaitu:

Menghapus kolom sensor_15 karena tidak memiliki nilai valid sama sekali.

Menghapus baris data yang masih mengandung nilai kosong (missing values) pada kolom sensor lainnya.

Alasan:
Keberadaan nilai hilang pada data sensor dapat mengganggu proses pembelajaran model dan menurunkan performa deteksi anomali. Penghapusan baris dengan nilai hilang dipilih untuk menjaga konsistensi data, mengingat jumlah data awal yang relatif besar.

### 3. Feature Selection

**Proses:**
Pemilihan fitur dilakukan dengan mengambil seluruh kolom sensor numerik yang tersedia dalam dataset, yaitu sensor_00 hingga sensor_51, serta mengabaikan kolom yang tidak relevan terhadap proses pemodelan. Kolom Unnamed: 0 dihapus karena hanya berfungsi sebagai indeks, sedangkan kolom timestamp digunakan untuk menjaga urutan temporal data namun tidak dimasukkan sebagai fitur input model. Selain itu, kolom machine_status digunakan sebagai label dalam proses klasifikasi.

Kolom sensor_15 dihapus karena seluruh nilainya kosong (missing values 100%).

**Alasan:**
Seluruh sensor dipertahankan sebagai fitur karena masing-masing merepresentasikan aspek kondisi operasional sistem pompa air. Menggunakan data sensor multivariat memungkinkan model mempelajari hubungan antar sensor secara simultan, yang penting dalam mendeteksi pola anomali yang kompleks.


### 4. Duplicate Handling

**Proses:**
Dilakukan pemeriksaan terhadap data duplikat menggunakan fungsi pendeteksian duplikasi pada dataset.

**Hasil:**
Tidak ditemukan data duplikat pada dataset.

**Alasan:**
Penghapusan data duplikat penting untuk mencegah bias pada model akibat pengulangan informasi yang sama dalam proses pelatihan.


### 5. Sliding Window Segmentation

**Proses:**
Data time series disegmentasi menggunakan pendekatan sliding window dengan ukuran jendela (window size) sebanyak 30 timestep dan pergeseran (stride) sebesar 1. Setiap window diberi label anomali apabila proporsi data anomali di dalam window mencapai atau melebihi 20%.

**Alasan:**
Pendekatan sliding window memungkinkan model menangkap pola temporal dan perubahan dinamika sensor dari waktu ke waktu. Penentuan label berdasarkan proporsi anomali dalam window membantu merepresentasikan kondisi sistem secara lebih kontekstual, bukan hanya berdasarkan satu titik waktu.

### 6. Data Splitting

**Proses:**
Dataset hasil segmentasi dibagi menjadi tiga bagian berdasarkan urutan waktu, yaitu:

70% data untuk pelatihan (training)

15% data untuk validasi (validation)

15% data untuk pengujian (testing)

**Alasan:**
Pembagian data secara berurutan dilakukan untuk menjaga sifat temporal data dan mencegah kebocoran informasi (data leakage) dari masa depan ke masa lalu.

### 7. Data Normalization

**Proses:**
Normalisasi data dilakukan menggunakan metode Min-Max Scaling, di mana setiap nilai sensor ditransformasikan ke dalam rentang [0, 1]. Proses normalisasi dilakukan dengan cara:

Melatih (fit) scaler hanya pada data training

Menerapkan (transform) scaler yang sama pada data validation dan testing

**Alasan:**
Sensor memiliki rentang nilai yang berbeda-beda. Normalisasi diperlukan untuk mencegah sensor dengan skala besar mendominasi proses pembelajaran model, serta untuk meningkatkan stabilitas dan konvergensi pelatihan jaringan saraf.

### 8. Class Imbalance Handling

**Proses:**
Karena distribusi kelas antara data normal dan anomali tidak seimbang, digunakan teknik class weighting pada tahap pelatihan model CNN–LSTM.

**Alasan:**
Penanganan ketidakseimbangan kelas diperlukan agar model tidak bias terhadap kelas mayoritas (normal) dan tetap mampu mengenali pola anomali secara efektif.


## Data Modelling

Pada tahap Data Modelling, dilakukan pembangunan dan pelatihan beberapa model klasifikasi untuk mendeteksi anomali pada data sensor IoT sistem pompa air. Tujuan dari tahap ini adalah untuk membandingkan performa model deep learning dengan model machine learning konvensional, serta menentukan model terbaik dalam mendeteksi kondisi anomali.

Dalam penelitian ini digunakan tiga model, yaitu Hybrid CNN–LSTM, Support Vector Machine (SVM), dan Random Forest.

1. Hybrid CNN–LSTM (Model Utama)

Hybrid CNN–LSTM merupakan model deep learning yang menggabungkan Convolutional Neural Network (CNN) dan Long Short-Term Memory (LSTM). CNN digunakan untuk mengekstraksi fitur spasial antar sensor dalam setiap window waktu, sedangkan LSTM digunakan untuk memodelkan ketergantungan temporal pada data time series.

**Arsitektur Model**

Model CNN–LSTM dibangun menggunakan beberapa lapisan sebagai berikut:

1. Convolutional Layer (Conv1D)

- Conv1D dengan 64 filter dan kernel size 3

- Fungsi aktivasi ReLU

- Padding same

- Regularisasi L2 untuk mengurangi overfitting

2. Batch Normalization & Max Pooling

- Batch normalization untuk menstabilkan proses pelatihan

- MaxPooling1D untuk reduksi dimensi fitur

3. Dropout Layer

- Dropout 0.3 digunakan untuk mencegah overfitting

4. Convolutional Layer Kedua

- Conv1D dengan 128 filter

- Konfigurasi serupa dengan layer pertama

5. LSTM Layer

- 64 unit LSTM

- Aktivasi tanh dan sigmoid

- Dropout dan recurrent dropout

- Return sequences diset False

6. Fully Connected Layer

- Dense 64 neuron dengan aktivasi ReLU

- Dropout 0.2

7. Output Layer

- Dense 1 neuron dengan aktivasi sigmoid untuk klasifikasi biner

**Parameter Utama Model CNN–LSTM**

- Conv1D filters: 64 dan 128

- Kernel size: 3

- LSTM units: 64

- Dropout: 0.2–0.3

- Regularization: L2 (1e-4)

- Output activation: Sigmoid

**Kelebihan CNN–LSTM**

- Mampu menangkap pola spasial dan temporal secara bersamaan

- Cocok untuk data sensor IoT multivariat

- Efektif untuk deteksi anomali berbasis urutan waktu

**Kekurangan CNN–LSTM**

- Membutuhkan komputasi yang lebih besar

- Sensitif terhadap pemilihan hyperparameter

2. Support Vector Machine (SVM)

Support Vector Machine merupakan algoritma klasifikasi yang bekerja dengan mencari hyperplane optimal untuk memisahkan dua kelas. Karena SVM tidak dapat langsung menerima input berbentuk time series, data hasil sliding window diratakan (flatten) terlebih dahulu menjadi vektor satu dimensi.


**Parameter Model SVM**

- kernel = 'rbf': 
Menggunakan kernel Radial Basis Function untuk menangkap pola non-linear

- C = 1.0:
Parameter regulasi untuk mengontrol trade-off antara margin dan kesalahan klasifikasi

- gamma = 'scale':
Menyesuaikan parameter kernel berdasarkan jumlah fitur

- class_weight = 'balanced':
Digunakan untuk menangani ketidakseimbangan kelas

**Kelebihan SVM**

- Efektif pada dataset berdimensi tinggi

- Mampu menangkap hubungan non-linear

**Kekurangan SVM**

- Tidak mempertimbangkan informasi temporal

- Kurang optimal untuk data time series kompleks

3. Random Forest

Random Forest adalah algoritma ensemble learning berbasis pohon keputusan yang menggabungkan banyak pohon untuk menghasilkan prediksi yang lebih stabil dan akurat. Seperti SVM, data time series juga diratakan sebelum digunakan sebagai input model.

**Parameter Model Random Forest**

- n_estimators = 200:
Jumlah pohon keputusan dalam hutan

- max_depth = None:
Tidak ada batasan kedalaman pohon

- min_samples_split = 2:
Jumlah minimum sampel untuk membagi node

- min_samples_leaf = 1:
Jumlah minimum sampel pada daun

- class_weight = 'balanced':
Mengatasi ketidakseimbangan kelas

- random_state = 42:
Digunakan untuk memastikan reprodusibilitas

- n_jobs = -1:
Menggunakan seluruh core CPU untuk pelatihan

**Kelebihan Random Forest**

- Tahan terhadap overfitting

- Dapat menangani data non-linear

- Mudah diinterpretasikan

**Kekurangan Random Forest**

- Tidak mempertimbangkan urutan waktu

- Kurang optimal untuk pola temporal jangka panjang



## Hyperparameter Tuning – Hybrid CNN–LSTM

Hyperparameter tuning dilakukan untuk memperoleh konfigurasi terbaik dari model Hybrid CNN–LSTM dalam mendeteksi anomali pada data sensor IoT sistem pompa air. Proses ini bertujuan untuk meminimalkan validation loss serta meningkatkan kemampuan generalisasi model terhadap data yang belum pernah dilihat.

Tuning dilakukan menggunakan Keras Tuner dengan metode Random Search.

**Tuning Objective**

- Objective: val_loss (minimization)

- Alasan: Validation loss dipilih untuk memastikan model tidak hanya baik pada data latih, tetapi juga mampu melakukan generalisasi dengan baik.

**Metode Tuning**

- Tuning Strategy: Random Search

- Library: Keras Tuner

- Jumlah Trial: 5

- Epoch per Trial: 20

- Batch Size: 64

Untuk mencegah overfitting, digunakan Early Stopping dengan konfigurasi:

- monitor = val_loss

- patience = 5

- restore_best_weights = True

**Model Architecture (Tuned)**

Model dibangun menggunakan arsitektur Hybrid CNN–LSTM yang terdiri dari:

- CNN untuk ekstraksi fitur spasial antar sensor

- LSTM untuk memodelkan dependensi temporal

- Dense layer untuk klasifikasi biner (Normal vs Anomaly)

Hyperparameter Search Space

| Layer / Block | Parameter | Value / Range | Keterangan |
| :--- | :--- | :--- | :--- |
| **🔹 CNN Block 1** | `filters_1` | 64 | Jumlah filter CNN |
| | `kernel_1` | 3 | Ukuran kernel |
| | `l2_1` | 1e-4 | Regularisasi L2 |
| | `dropout_1` | 0.2 – 0.5 | Pencegah overfitting |
| **🔹 CNN Block 2** | `filters_2` | 128 | Ekstraksi fitur tingkat lanjut |
| | `kernel_2` | 3 | Ukuran kernel |
| | `l2_2` | 1e-4 | Regularisasi L2 |
| | `dropout_2` | 0.2 – 0.5 | Pencegah overfitting |
| **🔹 LSTM Layer** | `lstm_units` | 64 | Jumlah unit LSTM |
| | `dropout_lstm` | 0.2 – 0.5 | Dropout temporal |
| | `l2_lstm` | 1e-4 | Regularisasi L2 |
| **🔹 Dense Layer** | `dense_units` | 64 | Jumlah neuron Dense layer |
| **🔹 Optimizer** | `learning_rate` | 1e-4, 3e-4, 1e-3 | Opsi laju pembelajaran |

Model terbaik hasil hyperparameter tuning memiliki konfigurasi sebagai berikut:

Conv1D (64 filters)

Batch Normalization

MaxPooling1D

Dropout

Conv1D (128 filters)

Batch Normalization

MaxPooling1D

Dropout

LSTM (64 units)

Dense (64 units)

Output Dense (1 unit, Sigmoid)

📊 Model Statistics

Total Parameters: 88,961

Trainable Parameters: 88,577

Non-trainable Parameters: 384

Model ini cukup ringan dan efisien, sehingga memungkinkan untuk diterapkan pada sistem monitoring berbasis IoT.

📈 Analysis & Insight

Kombinasi CNN + LSTM efektif dalam menangkap pola spasial dan temporal.

Dropout di kisaran 0.2 – 0.3 memberikan keseimbangan antara stabilitas dan generalisasi.

Learning rate 1e-3 memberikan konvergensi terbaik berdasarkan validation loss.

Random Search mampu menemukan konfigurasi optimal dengan jumlah trial yang relatif kecil.
