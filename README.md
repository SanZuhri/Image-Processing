# CitraPro: Toolkit Pengolahan Citra Digital Berbasis Web dengan FastAPI & OpenCV

*CitraPro* adalah aplikasi web pengolahan citra digital yang interaktif dan kaya fitur, dibangun menggunakan *FastAPI* untuk backend dan *OpenCV* sebagai mesin pemrosesan utamanya. Aplikasi ini menyediakan antarmuka yang modern dan mudah digunakan untuk melakukan berbagai macam operasi, mulai dari manipulasi dasar hingga analisis citra yang kompleks seperti analisis tekstur dan spektral.

Proyek ini dirancang sebagai platform edukasi yang kuat dan sebagai "playground" bagi para profesional atau mahasiswa yang ingin mengeksplorasi dunia pengolahan citra digital secara visual dan interaktif.

---

## ✨ Fitur Utama

Aplikasi ini memiliki antarmuka tab yang memisahkan alur kerja menjadi dua kategori utama:

#### 1. Pemrosesan 1 Input Citra (Single-Image Processing)

Kumpulan lengkap alat untuk menganalisis dan memanipulasi satu citra:

-   *Operasi Dasar*: Konversi Grayscale, Operasi Aritmatika (Penambahan, Pengurangan, dll.), dan Operasi Logika (NOT).
-   *Analisis Histogram*: Visualisasi histogram (Grayscale & RGB) dan Equalisasi Histogram.
-   *Filtering & Peningkatan Kualitas*: Berbagai jenis filter (Gaussian, Median, Bilateral), Penajaman Citra (Laplacian, Unsharp Masking), dan Padding.
-   *Manajemen Noise*: Penambahan noise (Salt & Pepper), Penghilangan noise (Denoising), dan Pengurangan Noise Periodik.
-   *Analisis Domain Frekuensi*: Transformasi Fourier untuk visualisasi spektrum magnitude dan phase.
-   *Analisis Citra*: Freeman Chain Code, Deteksi Tepi (Crack Code), dan Analisis Proyeksi Horizontal & Vertikal.
-   *Analisis Ruang Warna*: Konversi ke berbagai ruang warna (XYZ, LAB, HSV, dll.), Analisis per kanal, dan Analisis Luminance dengan data statistik.
-   *Analisis Tekstur*:
    -   *Statistik*: First-Order, GLCM, GLRLM.
    -   *Spektral*: Filter Gabor.
    -   *Struktural*: Local Binary Patterns (LBP) & Analisis Texton.
-   *Kompresi Citra*: Kompresi JPEG & PNG dengan level yang dapat diatur, lengkap dengan analisis rasio kompresi, PSNR, dan MSE.

#### 2. Pemrosesan 2 Input Citra (Double-Image Processing)

Operasi yang memerlukan dua citra input untuk perbandingan atau kombinasi:

-   *Operasi Logika*: Operasi bitwise AND, OR, dan XOR.
-   *Spesifikasi Histogram*: Menyesuaikan histogram citra sumber agar sesuai dengan histogram citra referensi (baik Grayscale maupun Warna).

---

## 🛠 Tumpukan Teknologi (Technology Stack)

-   *Backend*: FastAPI
-   *Pemrosesan Citra*: OpenCV-Python, NumPy, Scikit-Image
-   *Frontend*: Jinja2 Templates, HTML5, CSS3, JavaScript
-   *UI Framework*: Bootstrap 5
-   *Plotting*: Matplotlib

---

## 🚀 Instalasi & Menjalankan Proyek

1.  *Clone repositori:*
    bash
    git clone [URL-repositori-Anda]
    cd [nama-direktori-proyek]
    

2.  *Buat dan aktifkan virtual environment:*
    bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    

3.  *Install dependensi:*
    bash
    pip install -r requirements.txt
    

4.  *Jalankan aplikasi:*
    bash
    uvicorn main:app --reload
    

5.  Buka browser dan akses http://127.0.0.1:8000.