import streamlit as st
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

# Set up logging
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
    data = pd.read_csv('dataset/song_mood_data_expanded.csv')
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

# Streamlit app
st.title("Mood Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    file_path = os.path.join('uploads', uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    features = extract_features(file_path)
    if features is None:
        st.error("Feature extraction failed. Please try another file.")
    else:
        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        prediction = model.predict(features_scaled)[0]
        mood = label_encoder.inverse_transform([prediction])[0]
        probabilities = model.predict_proba(features_scaled)[0]
        mood_probs = {label_encoder.inverse_transform([i])[0]: f"{prob:.2%}" 
                      for i, prob in enumerate(probabilities)}
        
        train_accuracy, test_accuracy, overfit_status = evaluate_model_fit()
        
        st.write(f"**Predicted Mood:** {mood}")
        st.write(f"**Energy:** {features[0]:.4f}")
        st.write(f"**Valence:** {features[1]:.4f}")
        st.write(f"**Tempo:** {features[2]:.1f}")
        st.write(f"**Loudness:** {features[3]:.2f}")
        st.write("**Mood Probabilities:**")
        st.write(mood_probs)
        st.write(f"**Train Accuracy:** {train_accuracy}")
        st.write(f"**Test Accuracy:** {test_accuracy}")
        st.write(f"**Model Fit Status:** {overfit_status}")
        
        os.remove(file_path)

# Model evaluation section
st.header("Model Evaluation")
if st.button("Evaluate Model Fit"):
    train_accuracy, test_accuracy, overfit_status = evaluate_model_fit()
    st.write(f"**Train Accuracy:** {train_accuracy}")
    st.write(f"**Test Accuracy:** {test_accuracy}")
    st.write(f"**Model Fit Status:** {overfit_status}")

# Load and display model results
st.header("Model Results")
if st.button("Load Model Results"):
    try:
        with open('models/model_results.json', 'r') as f:
            results = st.json.load(f)
        st.write("**Model Results:**")
        st.write(results)
    except FileNotFoundError:
        st.error("Model results file not found. Please train the models first.")