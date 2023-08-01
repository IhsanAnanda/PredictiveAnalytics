# Proyek Machine Learning - Ihsan Ananda Pratama

## Domain Proyek

Susu adalah bahan pangan yang mengandung zat-zat nutrisi yang utama untuk kehidupan manusia, antara lain protein, lemak, karbohidrat, mineral, vitamin dan faktor-faktor pertumbuhan. Selain itu susu merupakan medium untuk beberapa mikroorganisme yang dapat merubah komposisi kimia susu selama penyimpanan **[1]**. Susu merupakan sumber nutrisi penting bagi tubuh manusia, terutama sebagai sumber kalsium, protein, vitamin D, vitamin B12, dan zat gizi lainnya. Memilih susu berkualitas baik memastikan Anda mendapatkan nutrisi yang optimal untuk menjaga kesehatan dan perkembangan tubuh.Susu yang berkualitas baik diproduksi dengan standar keamanan pangan yang ketat. Proses produksi yang memenuhi standar kebersihan dan keamanan akan mengurangi risiko terpapar mikroba berbahaya, bahan kimia beracun, atau kontaminan lainnya dalam susu. Susu berkualitas baik umumnya memiliki rasa yang segar, kaya, dan lezat. Proses produksi yang baik akan memastikan keaslian dan kesegaran rasa susu, serta konsistensi tekstur yang memuaskan.

Menurut laporan Unicef, jumlah penduduk yang menderita kekurangan gizi di dunia mencapai 767,9 juta orang pada 2021. Jumlah itu naik 6,4% dibandingkan pada tahun sebelumnya yang sebesar 721,7 juta orang **[2]**. Memilih susu berkualitas baik juga berarti mendukung praktik etis dalam peternakan susu. Produsen susu berkualitas biasanya memperhatikan kesejahteraan hewan, seperti memberikan makanan yang baik, tempat tinggal yang layak, dan perawatan yang memadai terhadap hewan-hewan tersebut. Pentingnya memilih susu berkualitas baik dapat menjaga kesehatan dan memenuhi standar gizi yang baik. Dengan memperhatikan faktor-faktor ini, Anda dapat memastikan bahwa susu yang Anda konsumsi memberikan manfaat terbaik bagi tubuh dan masyarakat secara keseluruhan.

## Business Understanding

Memilih susu dengan kualitas yang baik sangat penting untuk mengaja agar tubuh terpenuhi gizinya setiap hari agar terus terjaga kesehatan tubuh. Sebagai masyarakat ada berbagai cara umum yang dapat digunakan untuk mengidentifikasi kualitas susu yang baik tanpa menggunakan alat uji laboraturium. Kriteria yang dapat digunakan untuk mengidentifikasi kualitas susu yang baik secara umum adalah nilai pH, rasa, bau, tingkat kekeruhan, suhu, warna, dan lemak. Alat uji untuk menguji kriteria tersebut dapat dengan mudah didapatkan di pasaran dan tidak memerlukan pengetahuan yang menndalam, sehingga masyrakat dapat melakukannya di rumah dengan mudah.

### Problem Statements

Berdasarkan penjelasan sebelumnya maka rumusan masalah yang didapat adalah sebagai berikut:
- Dari kriteria yang telah disebutkan, apa kriteria yang paling berpengaruh dalam menentukan kualitas susu?
- Bagaimana kualitas susu yang baik berdasarkan kriteria yang telah disebutkan?
- Bagaimana cara membangun model machine learning yang dapat membantu dalam mengklasifikasi kualitas susu berdasarkan kriteria yang telah ditentukan?

### Goals

Untuk menyelesaikan rumusan masalah yang telah dijelaskan maka tujuan dari penelitian ini adalah sebagai berikut:
- Mengetahui kriteria susu yang paling berpengaruh dalam menentukan kualitas susu.
- Membangun model machine learning yang dapat mengidentifikasi kualitas susu berdasarkan kriteria yang telah disebutkan.
- Membangun model machine learning dengan metode _K-Nearest Neighbors_ dan _Random Forest_ kemudian memilih model terbaik berdasarkan hasil dari Mean Squared Error terendah.

## Data Understanding
Dataset yang digunakan pada proyek ini adalah dataset kualitas susu yang didapatkan dari Kaggle. Dataset ini terdiri dari 3 kelas kualitas susu dan 1059 baris data. Kualitas susu yang digunakan sebagai target kelas pada proyek ini memiliki 3 nilai yaitu _low_, _medium_, _high_. Kriteria atau fitur susu yang terdapat pada dataset ini adalah nilai pH, rasa, bau, tingkat kekeruhan, suhu, warna, dan lemak. Berikut ini pada Tabel 1 akan menampilkan contoh sample 5 baris data teratas dari dataset yang diperoleh. Sumber dataset : [Kaggle](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality).

Tabel 1. Tabel dataset yang diperoleh
|index|pH|Temprature|Taste|Odor|Fat |Turbidity|Colour|Grade|
|---|---|---|---|---|---|---|---|---|
|0|6\.6|35|1|0|1|0|254|high|
|1|6\.6|36|0|1|0|1|253|high|
|2|8\.5|70|1|1|1|1|246|low|
|3|9\.5|34|1|1|0|1|255|low|
|4|6\.6|37|0|0|0|0|255|medium|

### Variabel-variabel pada Milk Quality Prediction Kaggle adalah sebagai berikut:
- _Taste_ : merupakan rasa dari susu, jika bernilai 1 maka enak atau memenuhi standar dan akan bernilai 0  jika tidak enak. 
- _Odor_ : merupakan bau dari susu, jika bernilai 1 maka memenuhi standar atau tidak berbau busuk dan bernilai 0 jika berbau tidak enak atau busuk.
- _Fat_ : merupakan kadar lemak, jika bernilai 1 maka kadar lemak terpenuhi dan bernilai 0 jika kadar lemak tidak terpenuhi.
- _Turbidity_ : merupakan tingkat kekeruhan susu, jika bernilai 1 maka warna tingkat kekeruhan susu sesuai standar atau tidak terdapat senyawa lain dalam susu dan akan bernilai 0 jika tidak memenuhi standar.
- _Colour_ : merupakan warna susu dalam nilai hitam putih 0-255.
- pH : merupakan nilai pH susu.
- Temperature : merupakan nilai temperature susu ketika pertamakali dibuka dalam Fahrenheit.

![Pie Chart Image](https://github.com/IhsanAnanda/PredictiveAnalytics/blob/main/PieChart1.png?raw=true)
Gambar 1. Distribusi pembagian kelas dataset.
Berdasarkan visualisasi pada Gambar 1 yang menggunakan diagram lingkaran(pie chart) pada file notebook yang dilampirkan dari 1059 data terdapat 40,5%(429 data) dimiliki oleh kelas _low_, 35.3%(374 data) dimiliki oleh kelas _medium_, dan 24.2%(256 data) dimiliki oleh kelas _high_.

Tabel 2. Detail nilai setiap kolom dataset. 
|index|pH|Temprature|Taste|Odor|Fat |Turbidity|Colour|
|---|---|---|---|---|---|---|---|
|count|1059\.0|1059\.0|1059\.0|1059\.0|1059\.0|1059\.0|1059\.0|
|mean|6\.6301|44\.2266|0\.5467|0\.4325|0\.6714|0\.4910|251\.8404|
|std|1\.3997|10\.0984|0\.4980|0\.4956|0\.4699|0\.5002|4\.3074|
|min|3\.0|34\.0|0\.0|0\.0|0\.0|0\.0|240\.0|
|25%|6\.5|38\.0|0\.0|0\.0|0\.0|0\.0|250\.0|
|50%|6\.7|41\.0|1\.0|0\.0|1\.0|0\.0|255\.0|
|75%|6\.8|45\.0|1\.0|1\.0|1\.0|1\.0|255\.0|
|max|9\.5|90\.0|1\.0|1\.0|1\.0|1\.0|255\.0|

### Penjelasan nilai setiap variable:
- Pada kolom pHdapat dilihat bahwa rata-rata pH sebesar 6.6301 jarak kuartil pertama 6.5, jarak kuartil kedua 6.7, jarak kuartil ketiga 6.8, nilai minimum sebesar 3.0, dan nilai maksimum 9.5
- Pada kolom temprature dapat dilihat bahwa rata-rata suhu sebesar 44.2266, jarak kuartil pertama 38.0, jarak kuartil kedua 41.0, jarak kuartil ketiga 45.0, nilai minimum sebesar 34.0, dan nilai maksimum 90.0.
- Pada kolom colour dapat dilihat bahwa rata-rata warna sebesar 251.8404, jarak kuartil pertama 250.0, jarak kuartil kedua 255.0, jarak kuartil ketiga 255.0, nilai minimum sebesar 240.0, dan nilai maksimum 255.0.
- Pada kolom taste dapat dilihat bahwa rata-rata rasa sebesar 0.5467 yang menandakan bahwa rata-rata rasa susu memenuhi standar.
- Pada kolom odor dapat dilihat bahwa rata-rata bau sebesar 0.4325 yang menandakan bahwa rata-rata bau susu tidak memenuhi standar.
- Pada kolom turbidity dapat dilihat bahwa rata-rata kekeruhan sebesar 0.4910 yang menandakan bahwa rata-rata kekeruhan susu tidak memenuhi standar.
- Pada kolom fat dapat dilihat bahwa rata-rata lemak sebesar 0.6714 yang menandakan bahwa rata-rata rasa susu memenuhi standar.

## Data Preparation
Data preparation adalah proses mempersiapkan dan membersihkan data sebelum analisis. Ini melibatkan pembersihan data, transformasi, integrasi, pengurangan dimensi, dan validasi data untuk memastikan kualitas dan kesiapan data. Pada tahap ini data melalui beberapa proses sebagai berikut :
- Missing Values, pada tahapan ini dilakukan pengecekan apakah dataset yang diperoleh terdapat missing value atau data yang null. Setelah dilakukan pengecekan dapat disimpulkan bahwa data sudah lengkap dan tidak ada yang kosong, sehingga siap untuk dillakukan proses berikutnya.
- Normalisasi, pada tahapan ini setiap kolom numerical akan dilakukan normalisasi data menggunakan standar scaler. Tujuan dari tahapan ini adalah untuk merubah setiap nilai pada kolom numerical memiliki interval yang lebih dekat untuk mempermudah model dalam melakukan proses training.
- One-hot Encoding, pada tahapan ini fitur atau kolom yang bersifat kategorial akan diubah menjadi numerical agar dapat diproses oleh model. Kolom yang diubah tersebut antara lain : taste, odor, fat, dan turbidity.
- Train-test Split, pada tahapan ini dataset akan dibagi menjadi 2 yaitu sebagai data training dan data uji. Ratio perbandingan data training dengan data uji adalah 80:20, 847 data sebagai data training dan 212 sebagai data uji. 

## Model Development
Tahapan modelling dalam machine learning melibatkan pemilihan model yang sesuai, pemilihan fitur yang relevan, pemisahan data menjadi data latih dan data uji, pelatihan model menggunakan data latih, evaluasi kinerja model menggunakan data uji, dan penyetelan model untuk meningkatkan kinerja. Tujuan utama adalah mengembangkan model yang dapat mempelajari pola dari data latih dan menggeneralisasi dengan baik pada data baru untuk melakukan prediksi atau klasifikasi yang akurat. Berikut ini adalah algoritma machine learning yang digunakan sebagai model untuk mengklasifikasikan kualitas susu :
1. _K-Nearest Neighbors (KNN)_
	- Algoitma _KNN_ adalah algoritma yang menentukan kelas data yang diuji dengan cara mencari jumlah tetangga k terdekat atau paling mirip dengan data uji. Model ini tidak memerlukan proses training atau disebut dengan lazy learning yang dimana hanya akan melakukan kalkulasi jika akan melakukan klasifikasi. Model _KNN_ yang digunakan memiliki nilai k = 10.
	- Kelebihan : algoritma mudah dipahami, tidak memerlukan proses training, tidak memliki hyperparameter yang banyak hanya perlu menentukan nilai k dan rumus perhitungan jarak.
	- Kekurangan : akan menjadi semakin lambat jika memiliki data training yang banyak, sangat sensitif terhadap skala data, membutuhkan memori yang banyak ketika data training semakin banyak.
2. _Random Forest (RF)_
	- Algoritma _RF_ adalah sebuah metode penggabungan yang menggunakan sejumlah pohon keputusan _(decision trees)_ untuk melakukan klasifikasi atau regresi. Setiap pohon dalam _Random Forest_ diperoleh dengan menggunakan subsampling data secara acak dan pemilihan subset acak dari fitur-fitur yang tersedia. Parameter model yang digunakan yaitu maksimal kedalaman pohon _(level)_ adalah 16 dan maksimal jumlah pohon adalah 50.
	- Kelebihan : hasil output yang lebih konsisten, lebih tahan terhadap overfitting, akurasi yang lebih baik jika diberikan dengan dataset yang berukuran besar.
	- Kekurangan : beban komputasi yang berat, memerlukan proses training yang lama untuk pembentukan pohon, dan memiliki banyak hyperparameter yang perlu disesuaikan.
	
Kedua model tersebut dipilih karena merupakan algoritma yang populer dan sering digunakan untuk melakukan klasifikasi. Dalam studi kasus ini kedua model tersebut akan dilakukan berdasarkan matriks akurasi dan mengambil model dengan akurasi terbaik.

## Evaluation
Tahapan ini adalah tahapan mengevaluasi hasil training model dalam mengklasifikasikan data uji. Metrik evaluasi pada kasus klasifikasi kualitas susu ini adalah menggunakan metriks akurasi. Berikut ini adalah hasil tahapan evaluation pada kasus ini :
- Metrik akurasi adalah ukuran yang digunakan untuk mengevaluasi sejauh mana model klasifikasi mampu memprediksi dengan benar. Akurasi dihitung dengan membagi jumlah prediksi yang benar dengan total jumlah prediksi yang dilakukan.
- Hasil dari proses training dan pengujian kedua model adalah sebagai berikut : 

Tabel 3. Hasil perbandingan akurasi model.
|index|train|test|
|---|---|---|
|KNN|0\.9858323494687131|0\.9764150943396226|
|RF|1.0|1.0|

Dari Tabel 3 diatas dapat disimpulkan bahwa model dengan akurasi terbaik diperoleh oleh model dengan algoritma _Random Forest_. Metode tersebut mampu memperoleh akurasi sebesar 100%. Model KNN memperoleh akurasi yang tidak berbeda jauh dari model RF. Hal ini cukup mengejutkan karena model RF mampu memperoleh akurasi sebesar 100% baik untuk data training dan data uji.

## Kesimpulan
Kedua model yang dibandingkan dalam penelitian ini memliki kelebihan dan kekurangannya masing-masing. Meskipun model dengan algoritma _Random Forest_ memiliki akurasi lebih baik daripada model dengan algoritma _K-Nearest Neighbors_ juga memiliki akurasi yang lebih dari 95% yang dimana akurasi tersebut sudah sangat bagus. Namun melalui beberapa pertimbangan kelebihan dari setiap model, model dengan algoritma _Random Forest_ lebih baik untuk menangani dataset yang akan semakin bertambah. Maka hasil dari penelitian ini model dengan algoritma _Random Forest_ dipilih sebagai model terbaik. Model tersebut dapat diekspor dan diterapkan untuk membangun sebuah sistem klasifikasi kualitas susu untuk pengembangan lebih jauh.

## Referensi : 
[[1] R. Heti, "Kualitas Susu Pada Berbagai Pengolahan dan Penyimpanan," Balai Penelitian Ternak Bogor, hal. 497-502, 2020.](https://d1wqtxts1xzle7.cloudfront.net/57978703/SUSU_5-libre.pdf?1544620604=&response-content-disposition=inline%3B+filename%3DKUALITAS_SUSU_PADA_BERBAGAI_PENGOLAHAN_D.pdf&Expires=1686135989&Signature=E15D2D4sd3wxVVn4kNtIAhFRmVdg2fNzzrF5gx1uBmtj~qfA8H6LZZ3eVcb~PA7gol1EOO7CwbBdWgPKEnruyVCClPNJAGY68WDBNVI-TxxlZ6ujEczqWaAuM5nRc9Y0QXUkvFZ19a6LoQXJvQKdP7-6EE1HN7TmtQGkTncVdtvWL4~vm~mSh4IleAci8JRUEKvJUNJztCex0uF08fAvdaqANq6Bk4tMTdemD9pp-I6Dj-s~Yw~JSHwwmGu-UU1CyMwymvLVtUMORanWHzPbaGxONPRKROqxRL1aL92sgJ5km93vnR~LKq5D237S276Q8whp9q8rQ2R81yFo9Xiz5w__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
[[2] R. Monavia Ayu, "Unicef: 767,9 Juta Penduduk Dunia Menderita Kekurangan Gizi," DataIndonesia.id, 2022.](https://dataindonesia.id/ragam/detail/unicef-7679-juta-penduduk-dunia-menderita-kekurangan-gizi)

