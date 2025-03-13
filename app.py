import json
from flask import Flask, request, render_template, jsonify
import librosa
import numpy as np
import joblib
import os
import logging
import pandas as pd
import traceback
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from train_model import load_and_prepare_data, train_and_evaluate_models

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Load trained model, scaler, and label encoder
model = None
scaler = None
label_encoder = None
try:
    model = joblib.load('models/rf_mood_predictor.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/mood_label_encoder.pkl')
    logging.info("✅ Model, scaler, and label encoder loaded successfully!")
except FileNotFoundError:
    logging.warning("⚠️ Model, scaler, or label encoder not found. Please train the model first.")

# Load dataset for evaluation (optional)
X_train, X_test, y_train, y_test = None, None, None, None
try:
    data = pd.read_csv('song_mood_data_expanded.csv')
    if data['Valence'].dtype == 'object':
        data['Valence'] = data['Valence'].str.strip('[]').astype(float)
    X = data[['Energy', 'Valence', 'Tempo', 'Loudness']].values
    y = label_encoder.transform(data['Mood'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("✅ Dataset Loaded Successfully!")
except FileNotFoundError:
    logging.error("❌ Dataset file not found!")
except Exception as e:
    logging.error(f"❌ Error loading dataset: {e}")
    logging.error(traceback.format_exc())

# Evaluate model fit (overfit/underfit)
def evaluate_model_fit():
    if model is None or X_train is None or X_test is None:
        return "N/A", "N/A", "N/A"
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    accuracy_diff = train_accuracy - test_accuracy
    if accuracy_diff > 0.1:
        overfit_status = "Overfit"
    elif train_accuracy < 0.7:
        overfit_status = "Underfit"
    else:
        overfit_status = "Good Fit"
    
    return f"{train_accuracy:.2%}", f"{test_accuracy:.2%}", overfit_status

# Enhanced feature extraction
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        valence = float(np.mean(chroma) * np.mean(spectral_centroid) / 1000)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo) if tempo else 0.0
        loudness = float(20 * np.log10(np.mean(rms) + 1e-10))
        features = [energy, valence, tempo, loudness]
        logging.info(f"Extracted features: {features}")
        return features
    except Exception as e:
        logging.error(f"❌ Feature extraction failed: {e}")
        logging.error(traceback.format_exc())
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No selected file"}), 400

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    features = extract_features(file_path)
    if features is None:
        os.remove(file_path)
        return jsonify({"error": "Feature extraction failed"}), 500

    features_scaled = scaler.transform(np.array(features).reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    mood = label_encoder.inverse_transform([prediction])[0]
    probabilities = model.predict_proba(features_scaled)[0]
    mood_probs = {label_encoder.inverse_transform([i])[0]: f"{prob:.2%}" 
                  for i, prob in enumerate(probabilities)}

    train_accuracy, test_accuracy, overfit_status = evaluate_model_fit()
    os.remove(file_path)

    return render_template('result.html',
                          mood=mood,
                          energy=f"{features[0]:.4f}",
                          valence=f"{features[1]:.4f}",
                          tempo=f"{features[2]:.1f}",
                          loudness=f"{features[3]:.2f}",
                          probabilities=mood_probs,
                          train_accuracy=train_accuracy,
                          test_accuracy=test_accuracy,
                          overfit_status=overfit_status)

@app.route('/models')
def show_models():
    # Load saved model results from JSON file
    try:
        with open('models/model_results.json', 'r') as f:
            results = json.load(f)
        logging.info("✅ Model results loaded successfully!")
    except FileNotFoundError:
        logging.error("❌ Model results file not found. Please train the models first.")
        results = {}

    return render_template('models.html', results=results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
