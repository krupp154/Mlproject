import streamlit as st
import json
import librosa
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Load trained model, scaler, and label encoder
model = joblib.load('models/rf_mood_predictor.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/mood_label_encoder.pkl')

# Load dataset for evaluation (optional)
try:
    data = pd.read_csv('dataset/song_mood_data_expanded.csv')
    if data['Valence'].dtype == 'object':
        data['Valence'] = data['Valence'].str.strip('[]').astype(float)
    X = data[['Energy', 'Valence', 'Tempo', 'Loudness']].values
    y = label_encoder.transform(data['Mood'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception:
    X_train, X_test, y_train, y_test = None, None, None, None

def evaluate_model_fit():
    if model is None or X_train is None or X_test is None:
        return "N/A", "N/A", "N/A"
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    accuracy_diff = train_accuracy - test_accuracy
    if accuracy_diff > 0.1:
        overfit_status = "Overfit"
    elif train_accuracy < 0.7:
        overfit_status = "Underfit"
    else:
        overfit_status = "Good Fit"

    return f"{train_accuracy:.2%}", f"{test_accuracy:.2%}", overfit_status

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    valence = float(np.mean(chroma) * np.mean(spectral_centroid) / 1000)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if tempo else 0.0
    loudness = float(20 * np.log10(np.mean(rms) + 1e-10))
    return [energy, valence, tempo, loudness]

# Streamlit UI
st.title("🎵 Mood Prediction App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload", "Models"])

if page == "Home":
    st.markdown("Welcome to the **Mood Prediction App**! Upload a music file to predict its mood.")

elif page == "Upload":
    uploaded_file = st.file_uploader("Upload a music file", type=["wav", "mp3"])
    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        features = extract_features(file_path)
        if features:
            features_scaled = scaler.transform(np.array(features).reshape(1, -1))
            prediction = model.predict(features_scaled)[0]
            mood = label_encoder.inverse_transform([prediction])[0]
            probabilities = model.predict_proba(features_scaled)[0]
            mood_probs = {label_encoder.inverse_transform([i])[0]: f"{prob:.2%}" 
                          for i, prob in enumerate(probabilities)}

            st.success(f"🎧 Predicted Mood: **{mood}**")
            st.json(mood_probs)

            train_accuracy, test_accuracy, overfit_status = evaluate_model_fit()
            st.markdown(f"**Training Accuracy:** {train_accuracy}")
            st.markdown(f"**Test Accuracy:** {test_accuracy}")
            st.markdown(f"**Model Fit Status:** {overfit_status}")
        else:
            st.error("❌ Feature extraction failed. Please try again.")

elif page == "Models":
    try:
        with open('models/model_results.json', 'r') as f:
            results = json.load(f)
        st.json(results)
    except FileNotFoundError:
        st.error("❌ Model results file not found. Please train the models first.")