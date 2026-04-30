import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing.feature_engineering import FeatureEngineer, get_labels

def train_and_save_models():
    # Load dummy data
    data_path = os.path.join('data', 'meeting_report.csv')
    if not os.path.exists(data_path):
        print("Data not found. Run generate_dummy_data.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Feature engineering
    fe = FeatureEngineer()
    X = fe.preprocess(df)
    X_scaled = fe.scale_features(X, fit=True)
    
    # Get labels for supervised learning
    y_multi, y_binary = get_labels(df)
    
    # 1. Random Forest (Multi-class)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_multi)
    
    # 2. SVM (Binary)
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_scaled, y_binary)
    
    # 3. KNN (Similarity)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y_multi) # Using multi-class labels for KNN
    
    # 4. Naive Bayes (Lightweight)
    nb = GaussianNB()
    nb.fit(X_scaled, y_multi)
    
    # 5. K-Means (Clustering)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Save models and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/random_forest.joblib')
    joblib.dump(svm, 'models/svm.joblib')
    joblib.dump(knn, 'models/knn.joblib')
    joblib.dump(nb, 'models/naive_bayes.joblib')
    joblib.dump(kmeans, 'models/kmeans.joblib')
    joblib.dump(fe.scaler, 'models/scaler.joblib')
    
    print("Models and scaler saved to models/")
    
    # Evaluation
    y_pred_rf = rf.predict(X_scaled)
    print("\nRandom Forest Accuracy:", accuracy_score(y_multi, y_pred_rf))
    print("\nClassification Report (RF):\n", classification_report(y_multi, y_pred_rf))

if __name__ == "__main__":
    train_and_save_models()
