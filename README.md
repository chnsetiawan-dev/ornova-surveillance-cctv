# Ornova Surveillance CCTV — deteksi APD & zona bahaya

Prototipe aplikasi pemantauan keselamatan area dari video CCTV: dashboard web + server deteksi YOLO lokal.

- **Kepatuhan APD per area** — tiap area punya daftar APD wajib (helm, masker, sepatu, dll.).
- **Warna helm** — helm wajib putih, kuning, atau merah (bisa diubah).
- **Intrusi zona** — gambar kotak zona terlarang di atas video; orang yang kakinya masuk dicatat.
- **Orang dekat kendaraan** (opsional per area) — checklist "Alarm orang dekat kendaraan".
- **Kemungkinan terjatuh** — orang dalam posisi rebah.
- Log kejadian dengan foto bukti, ekspor CSV, dan mode simulasi untuk area yang belum punya video.

## Menjalankan

1. Pasang Python 3.11 dan paket (dari jendela **cmd biasa**):
   ```
   py -3 -m pip install --user -r requirements.txt
   ```
   Untuk GPU NVIDIA pasang torch versi CUDA, mis.:
   ```
   py -3 -m pip install --user --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
2. Klik dua kali `Jalankan Surveillance.bat`. Bobot model diunduh otomatis (`app/download_models.py`) lalu
   dashboard terbuka di http://localhost:8000. Jangan tutup jendela server.
3. Pilih area → pilih file video → **Unggah & deteksi**. **Gambar zona** untuk menandai zona terlarang.

> Jika paket dipasang dari dalam aplikasi Claude desktop, paket bisa tersimpan di folder kontainer aplikasi
> (`%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming`) dan tidak terlihat saat `.bat` diklik dua kali
> (`ModuleNotFoundError`). Pasang dari cmd biasa.

## Menyesuaikan dengan lokasi

- **Daftar area** ada di `dashboard.html` (`const A=[...]`): nomor, nama, tingkat risiko (0 bebas, 1 sedang,
  2 tinggi), daftar bahaya, APD wajib, dan `true` bila ada lalu lintas forklift. Isi di repo ini hanya contoh.
- **Ambang & aturan deteksi** ada di `CFG` di `app/server.py` (confidence, `min_head_px`, `window_s`,
  `helmet_colors_allowed`, `vehicle_area_cooldown_s`, dll.).

## Model yang dipakai

| Fungsi | Model |
|---|---|
| Orang, kendaraan, tracking | Ultralytics YOLOv8n (COCO) + ByteTrack |
| Helm | `keremberke/yolov8s-hard-hat-detection` (Hardhat / NO-Hardhat) |
| APD lain (masker, sarung tangan, kacamata, sepatu) | `keremberke/yolov8m-protective-equipment-detection` |

## Batasan (prototipe)

- Model APD publik: helm cukup baik; masker, sarung tangan, kacamata, sepatu masih lemah. Earplug, apron,
  jas lab, jaket, kedok las, jas hujan, seragam belum terdeteksi — perlu pelatihan dengan footage lokasi.
- Topi, kupluk, dan helm motor kadang terbaca sebagai helm.
- Forklift/alat berat tidak dikenali (proksi kelas mobil/motor/truk); jarak dalam rasio frame, bukan meter.
- Kecepatan: ±1 fps di CPU, ±20–30 fps di GPU RTX 4050 per video.
