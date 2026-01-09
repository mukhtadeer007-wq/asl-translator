from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os, re, base64, io, uuid, time
from PIL import Image
import numpy as np
import mediapipe as mp
import joblib
from gtts import gTTS

app = Flask(__name__, static_folder='static', template_folder='templates')

MODEL_PATH = os.path.join('model', 'asl_model.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

classifier = None
if os.path.exists(MODEL_PATH):
    try:
        classifier = joblib.load(MODEL_PATH)
        print('Loaded model from', MODEL_PATH)
    except Exception as e:
        print('Failed to load model:', e)

def base64_to_pil(img_b64: str):
    header_re = re.compile(r'^data:image/.+;base64,')
    img_b64 = header_re.sub('', img_b64)
    img_bytes = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(img_bytes)).convert('RGB')

def extract_hand_landmarks_from_pil(img: Image.Image):
    img_np = np.array(img)
    # MediaPipe expects RGB images
    results = hands.process(img_np)
    if not results.multi_hand_landmarks:
        return None
    landmarks = results.multi_hand_landmarks[0]
    pts = []
    for lm in landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.z])
    return np.array(pts)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global classifier
    data = request.json
    if 'image' not in data:
        return jsonify({'success': False, 'error': 'No image provided.'}), 400
    try:
        pil = base64_to_pil(data['image'])
    except Exception as e:
        return jsonify({'success': False, 'error': f'Invalid image: {e}'}), 400

    features = extract_hand_landmarks_from_pil(pil)
    if features is None:
        return jsonify({'success': False, 'error': 'No hand detected.'}), 200

    if classifier is not None:
        try:
            pred = classifier.predict([features])[0]
            prob = None
            if hasattr(classifier, 'predict_proba'):
                prob = float(np.max(classifier.predict_proba([features])))
            # Generate speech using gTTS
            try:
                speech_id = str(uuid.uuid4()) + '.mp3'
                speech_dir = os.path.join('static', 'speech')
                os.makedirs(speech_dir, exist_ok=True)
                tts = gTTS(text=str(pred), lang='en')
                speech_path = os.path.join(speech_dir, speech_id)
                tts.save(speech_path)
                speech_url = url_for('static', filename='speech/' + speech_id) + '?t={}'.format(int(time.time()))
            except Exception as e:
                speech_url = None
            return jsonify({'success': True, 'letter': str(pred), 'confidence': prob, 'speech_url': speech_url}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': f'Prediction failed: {e}'}), 500
    else:
        return jsonify({'success': True, 'letter': None, 'landmarks': features.tolist(), 'note': 'No classifier found. Run train_model.py to create model/asl_model.pkl'}), 200

@app.route('/download-model')
def download_model():
    if os.path.exists(MODEL_PATH):
        return send_from_directory('model', 'asl_model.pkl', as_attachment=True)
    else:
        return jsonify({'success': False, 'error': 'No model available.'}), 404

if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    os.makedirs(os.path.join('static','speech'), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
