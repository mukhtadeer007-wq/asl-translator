# ASL Translator (Flask) — Demo Package

This package contains a Flask-based starter web app that demonstrates converting ASL (images) to text and spoken audio (using MediaPipe for landmarks + optional classifier + gTTS).

## Contents
- `app.py` — Flask app that extracts MediaPipe hand landmarks and (if `model/asl_model.pkl` exists) predicts a letter. It also generates speech mp3s using gTTS and serves them from `static/speech/`.
- `train_model.py` — script that extracts landmarks from `dataset/<LETTER>/*.png` and trains a RandomForest classifier.
- `templates/index.html`, `static/script.js` — frontend UI, captures webcam frames and plays returned speech audio.
- `requirements.txt` — dependencies.
- `dataset/` — **synthetic demo dataset** A–Z (10 images per letter). These images are simple letter images (not real hand photos), intended to let you test the training pipeline structure. For real accuracy, replace these with real hand images.
- `model/` — empty; training will write `model/asl_model.pkl`.

## Quick start
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate    # Windows
pip install -r requirements.txt
```

2. (Optional) Train model:
```bash
python train_model.py
```
Note: The included synthetic images are NOT real hand photos — MediaPipe may fail to detect hands in them. Replace `dataset/` with a real ASL image set for training.

3. Run the app:
```bash
python app.py
```
Open http://localhost:5000 and allow camera access. Click *Capture & Predict* to send a frame to the server. If a model is available and a prediction is returned, the server will generate an audio mp3 using gTTS and the browser will play it.

## Notes
- The package includes a lightweight synthetic dataset for demo/testing folder structure only. For meaningful results replace `dataset/` with real hand images (Kaggle has ASL alphabet datasets).
- gTTS requires internet access at runtime to generate audio.
