# Real-time Age & Gender Detection (OpenCV + CNN)

This project uses OpenCV's DNN module with pre-trained CNN models to detect faces and estimate gender and age in real time from your webcam.

## Features
- Face detection via OpenCV DNN (ResNet SSD) with fallback to Haar cascades.
- Age and Gender classification via Caffe models (AgeNet/GenderNet).
- Works on Windows with Python 3.9+.

## Setup
1. Create a virtual environment (recommended) and install dependencies:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Download models (face, age, gender):
```bash
python scripts/download_models.py
```

The face detector comes from official OpenCV sources and should download reliably. The community-hosted Age/Gender models sometimes move; the downloader tries multiple mirrors. If age/gender files fail to download, use the manual links below and place them into the indicated folders:

- `models/age/age_deploy.prototxt`
- `models/age/age_net.caffemodel`
- `models/gender/gender_deploy.prototxt`
- `models/gender/gender_net.caffemodel`

Manual sources (if the script fails):
- LearnOpenCV Age/Gender models repository.
- Alternative community mirrors hosting the same files.

3. Run the real-time app:
```bash
python scripts/realtime_age_gender.py
```

Press `q` to quit.

### Options and overlays
- The preview is mirrored for a natural webcam feel.
- Overlays include:
  - Gender with probability
  - Face detection confidence
  - Age range with probability
- FPS is displayed on the top-left.

### Temporal smoothing
- We apply EMA smoothing to probabilities so the labels are steadier frame-to-frame.
- Tune smoothing via env var `SMOOTH_BETA` (default 0.9). Lower values react faster; higher values are smoother.

### Display size and camera options
- Resize the on-screen window using `--scale` (or set env `DISPLAY_SCALE`):
```bash
python scripts/realtime_age_gender.py --scale 1.5
```
- Request a specific camera resolution:
```bash
python scripts/realtime_age_gender.py --cam-width 1280 --cam-height 720
```
- Force a camera index if you have multiple cameras:
```bash
python scripts/realtime_age_gender.py --cam-index 1
```

### Troubleshooting alignment
If labels overlap your hair/edge of frame, the app now packs labels above the box when there is room; otherwise inside or below the box to avoid clashing.

## Notes
- Age is predicted as one of 8 ranges: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100).
- Mean subtraction BGR values for Caffe Age/Gender nets: [78.426, 87.769, 114.896].
- If no DNN face model is found, the app falls back to Haar cascades.

## Troubleshooting
- If the webcam does not open, ensure no other app is using it and try device index 1 or 2.
- If models fail to download, place them manually in `models/face`, `models/age`, and `models/gender` as listed above.
- For best FPS, run on a machine with a recent CPU and use a smaller camera resolution.


