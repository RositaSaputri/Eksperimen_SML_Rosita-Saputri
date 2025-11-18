# Eksperimen SML – Rosita Saputri

Repositori ini berisi proses Exploratory Data Analysis (EDA) dan preprocessing
dataset stroke sebagai bagian dari penyelesaian Proyek Akhir Kriteria 1
pada kelas "Belajar Pengembangan Machine Learning" – Dicoding.

## 🔹 Struktur Folder
Eksperimen_SML_Rosita-Saputri/
│
├── preprocessing/
│ ├── data-stroke.csv → Dataset mentah
│ ├── stroke_preprocessed.zip → Hasil preprocessing dalam format ZIP
│ ├── automate_Rosita.py → Script otomatisasi preprocessing
│ └── Eksperimen_RositaSaputri.ipynb → Notebook EDA + preprocessing
│
└── requirements.txt


## 🔹 Penjelasan File

### **1. data-stroke.csv**
Dataset mentah yang digunakan dalam eksperimen.

### **2. stroke_preprocessed.zip**
Berisi file `stroke_preprocessed.csv`, yaitu hasil preprocessing akhir.  
File ini dikompresi menjadi ZIP karena ukuran lebih besar dan untuk mempermudah upload.

Reviewer dapat mendownload ZIP ini dan mengekstraknya untuk melihat hasil preprocessing.

### **3. automate_Rosita.py**
Skrip Python untuk menjalankan proses preprocessing secara otomatis.

### **4. Eksperimen_RositaSaputri.ipynb**
Notebook berisi:
- Perkenalan dataset
- Import library
- Exploratory Data Analysis (EDA)
- Data preprocessing lengkap

### **5. requirements.txt**
Daftar library yang diperlukan untuk menjalankan notebook dan script:
pandas
numpy
matplotlib
seaborn
scikit-learn


## 🔹 Cara Menjalankan Script Preprocessing

1. Pastikan seluruh dependency sudah diinstall:
pip install -r requirements.txt

2. Jalankan script otomatisasi:
python automate_Rosita.py

Hasil preprocessing akan tersimpan di folder `preprocessing`.

## 🔹 Catatan Penting
- File hasil preprocessing disimpan sebagai **ZIP** (`stroke_preprocessed.zip`) untuk
  memudahkan upload ke GitHub.
- Isi ZIP tersebut adalah file `stroke_preprocessed.csv`.

Jika ada yang kurang atau perlu diperbaiki, silakan memberi tahu saya.
