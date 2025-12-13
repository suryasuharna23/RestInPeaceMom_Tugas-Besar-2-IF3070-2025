# Tugas Besar 2 - IF3070 Dasar Artificial Intelligence
## Kelompok 12
### Fraud Detection using Decision Tree, Logistic Regression, and K Nearest Neighbor Machine Learning Algorithm

Repository ini berisi implementasi berbagai algoritma pembelajaran mesin yaitu Decision Tree, Logistic Regression, and K Nearest Neighbor from scratch. Kemudian model diujikan pada suatu dataset melalui platform Kaggle Competition. Performa model juga akan dibandingkan dengan model yang dibuat dari library _scikit-learn_. Proyek ini dibuat untuk memenuhi Tugas Besar 2 mata kuliah IF3070 Dasar Inteligensi Artifisial Semester 1 Tahun Ajaran 2025/2026.

* **CART Decision Tree**
* **Logistic Regression**
* **K Nearest Neighbor**

---

## Setup & Instalasi

### 1. Prasyarat
* **Python**
* **Pip**
* **Jupyter Notebook**

### 2. Instalasi Dependensi
1.  Clone repository:
    ```bash
    git clone https://github.com/suryasuharna23/RestInPeaceMom_Tugas-Besar-2-IF3070-2025.git
    ```

2.  Install dependensi yang diperlukan pada `requirements.txt`:
    ```bash
    pip install -r src/requirements.txt
    ```

---

## Cara Menjalankan Program

Cara untuk menjalankan program, yaitu melalui **Jupyter Notebook**.

### Melalui Jupyter Notebook
Notebook ini berisi alur kerja lengkap mulai dari pemrosesan data, pelatihan model, hingga evaluasi dan prediksi.

1. Buka Jupyter Notebook:
   ```bash
   jupyter notebook

2. Buka file: src/Kelompok 12_ IF3070 Dasar Artificial Intelligence _ Tugas Besar 2 Notebook Template.ipynb

3. Jalankan seluruh sel secara berurutan, atau
4. Jalankan sel yang dibutuhkan
---

Model dapat di-save dan di-load menggunakan fungsi save_model dan load_model.
Model yang digunakan pada submisi kaggle dapat diperoleh dengan menjalankan algoritma DTL dan Logress pada settingan hyperparameter bawaan.

## Anggota Kelompok dan Pembagian Tugas
| Nama Anggota | NIM | Pembagian Tugas |
|---|---| ---|
| Muhammad Aymar Barkhaya | 18223051 | 1. Implementasi DTL from scratch dan optimisasi <br> 2. Save dan load model <br> 3. Spam Kaggle submission<br> 4. Mengerjakan dokumen laporan <br> 5. Membuat README.MD |
| Surya Suharna | 18223075 | 1. Implementasi Logistic Regression from scratch dan optimisasi <br> 2. Mengerjakan bonus gambar visualisasi DTL <br> 3. Mengerjakan bonus video garis kontur fungsi loss <br> 4. Spam Kaggle submission <br> 5. Mengerjakan dokumen laporan |
| Ni Made Sekar Jelita Parameswari | 18223101 | 1. Data cleaning <br> 2. Data preprocessing <br> 3. Pipelining data and algorithm <br> 4. Spam Kaggle submission <br> 5. Mengerjakan dokumen laporan |
| Muhammad Azzam Robbani | 18223025 | 1. Implementasi KNN from scratch <br> 2. Mengerjakan bonus video proses training model KNN <br> 3. Perbandingan algoritma from scratch dengan pustaka _scikit-learn_ <br> 4.Mengerjakan dokumen laporan |
---
