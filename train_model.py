#\"\"\"train_model.py
#Train a simple RandomForest on MediaPipe landmarks from dataset/<LABEL>/*.png
#\"\"\"
import os
from glob import glob
from tqdm import tqdm
import mediapipe as mp
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

DATASET_DIR = 'dataset'  # expected structure: dataset/A/*.png

def extract_landmarks_from_image(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        return None
    img_np = np.array(img)
    results = hands.process(img_np)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    pts = []
    for p in lm.landmark:
        pts.extend([p.x, p.y, p.z])
    return np.array(pts)

X = []
Y = []

if not os.path.exists(DATASET_DIR):
    print('Dataset directory not found:', DATASET_DIR)
    print('Create dataset/<LABEL> and add images for each letter.')
    exit(1)

labels = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
if not labels:
    print('No labeled subfolders found under', DATASET_DIR)
    exit(1)

for label in labels:
    files = glob(os.path.join(DATASET_DIR, label, '*'))
    for f in tqdm(files, desc=f'Processing {label}'):
        feats = extract_landmarks_from_image(f)
        if feats is None:
            continue
        X.append(feats)
        Y.append(label)

if not X:
    print('No samples with detected hands were found. Ensure images show hands and are compatible with MediaPipe.')
    exit(1)

X = np.array(X)
Y = np.array(Y)

print('Extracted', X.shape[0], 'samples with', X.shape[1], 'features')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=Y)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(classification_report(y_test, pred))

os.makedirs('model', exist_ok=True)
joblib.dump(clf, os.path.join('model', 'asl_model.pkl'))
print('Saved model to model/asl_model.pkl')
