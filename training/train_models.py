import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing.feature_engineering import FeatureEngineer, get_labels, FEATURE_COLUMNS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_save_models():
    # Load data
    data_path = os.path.join('data', 'meeting_report.csv')
    if not os.path.exists(data_path):
        logger.error("Data not found. Run generate_dummy_data.py first.")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records for training.")
    
    # Feature engineering
    fe = FeatureEngineer()
    X = fe.preprocess(df)
    
    # Ensure consistent feature order
    X = X[FEATURE_COLUMNS]
    X_scaled = fe.scale_features(X, fit=True)
    
    # Get labels for supervised learning
    y_multi, y_binary = get_labels(df)
    
    models = {}
    
    # 1. Random Forest (Multi-class)
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_multi)
    models['random_forest'] = rf
    
    # 2. SVM (Binary)
    logger.info("Training SVM...")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_scaled, y_binary)
    models['svm'] = svm
    
    # 3. KNN (Similarity)
    logger.info("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y_multi)
    models['knn'] = knn
    
    # 4. Naive Bayes (Lightweight)
    logger.info("Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_scaled, y_multi)
    models['naive_bayes'] = nb
    
    # 5. K-Means (Clustering)
    logger.info("Training K-Means...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    models['kmeans'] = kmeans
    
    # Save models and scaler
    os.makedirs('models', exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f'models/{name}.joblib')
    joblib.dump(fe.scaler, 'models/scaler.joblib')
    
    logger.info("Models and scaler saved to models/")
    
    # Evaluation
    evaluation_results = []
    evaluation_results.append("=== Model Evaluation Report ===\n")
    
    # Evaluate Random Forest
    y_pred_rf = rf.predict(X_scaled)
    evaluation_results.append("Random Forest (Engagement Level):")
    evaluation_results.append(f"Accuracy: {accuracy_score(y_multi, y_pred_rf):.4f}")
    evaluation_results.append("Classification Report:")
    evaluation_results.append(classification_report(y_multi, y_pred_rf))
    evaluation_results.append("Confusion Matrix:")
    evaluation_results.append(str(confusion_matrix(y_multi, y_pred_rf)))
    evaluation_results.append("-" * 30 + "\n")
    
    # Evaluate SVM
    y_pred_svm = svm.predict(X_scaled)
    evaluation_results.append("SVM (Binary Status):")
    evaluation_results.append(f"Accuracy: {accuracy_score(y_binary, y_pred_svm):.4f}")
    evaluation_results.append("-" * 30 + "\n")
    
    # Save evaluation to file
    report_path = 'models/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("\n".join(evaluation_results))
    
    logger.info(f"Evaluation report saved to {report_path}")
    print("\n".join(evaluation_results))

if __name__ == "__main__":
    train_and_save_models()


if __name__ == "__main__":
    train_and_save_models()
