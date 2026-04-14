# Proyek Analisis Data

## Struktur Folder
- `dashboard/`: berkas dashboard streamlit dan data olahan untuk visualisasi.
- `data/`: dataset yang digunakan untuk analisis.
- `notebook.ipynb`: notebook analisis data.
- `requirements.txt`: daftar library.
- `url.txt`: tautan dashboard yang sudah dideploy.

## Cara Menjalankan Notebook
1. Pastikan Python sudah terpasang.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Buka dan jalankan `notebook.ipynb` secara berurutan.

## Cara Menjalankan Dashboard
1. Masuk ke folder proyek `submission`.
2. Jalankan perintah berikut:
   ```bash
   streamlit run dashboard/dashboard.py
   ```
3. Dashboard akan terbuka otomatis di browser.

## Deploy ke Streamlit Community Cloud
1. Upload folder proyek ke repository GitHub.
2. Buka Streamlit Community Cloud dan login menggunakan akun GitHub.
3. Pilih repository proyek ini.
4. Atur `Main file path` menjadi `submission/dashboard/dashboard.py`.
5. Klik `Deploy`.
6. Setelah berhasil deploy, salin URL aplikasi dan tulis ke `submission/url.txt`.

## Catatan Data Dashboard
- Dashboard membaca data dari folder `submission/data/`.
- Berkas `submission/dashboard/main_data.csv` disediakan sebagai data olahan tambahan untuk kebutuhan eksplorasi dashboard.