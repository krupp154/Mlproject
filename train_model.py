import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os
import logging
import json  # Add this import

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

def load_and_prepare_data(file_path):
    """Load and preprocess the dataset"""
    try:
        data = pd.read_csv(file_path)
        if data['Valence'].dtype == 'object':
            data['Valence'] = data['Valence'].str.strip('[]').astype(float)
        X = data[['Energy', 'Valence', 'Tempo', 'Loudness']].values
        y = data['Mood'].values
        logging.info("✅ Data loaded and prepared successfully!")
        return X, y
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        raise

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models with cross-validation"""
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models to test
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel='linear', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    # Train and evaluate each model with cross-validation
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        logging.info(f"🚀 Training {model_name} with cross-validation...")

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

        # Train the model on the full dataset for final evaluation
        model.fit(X_train_scaled, y_train)

        # Calculate test accuracy
        test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

        # Store results
        results[model_name] = {
            "cv_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": test_accuracy
        }

        # Save the Random Forest model (assuming it's used in app.py)
        if model_name == "Random Forest":
            joblib.dump(model, 'models/rf_mood_predictor.pkl')
            joblib.dump(scaler, 'models/scaler.pkl')
            joblib.dump(label_encoder, 'models/mood_label_encoder.pkl')
            logging.info("✅ Random Forest saved for app use!")

    # Save results to JSON file for faster loading in the app
    with open('models/model_results.json', 'w') as f:
        json.dump(results, f)

    logging.info("✅ Model results saved successfully!")

    # Print accuracy comparison
    print("\nModel Accuracy Comparison:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Cross-Validation Accuracy: {metrics['cv_accuracy']:.4f} (±{metrics['cv_std']:.4f})")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")

    return results
if __name__ == '__main__':
    data_file = 'song_mood_data_expanded.csv'
    X, y = load_and_prepare_data(data_file)
    train_and_evaluate_models(X, y)