import pandas as pd
import numpy as np
import joblib
import os
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, silhouette_score
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
    
    # Split data into Training and Testing
    X_train, X_test, y_train_multi, y_test_multi = train_test_split(
        X_scaled, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    models = {}
    results = {}

    # 1. Random Forest (Multi-class)
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train_multi)
    models['random_forest'] = rf
    
    # 2. SVM (Binary)
    logger.info("Training SVM...")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_bin, y_train_bin)
    models['svm'] = svm
    
    # 3. KNN (Similarity)
    logger.info("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train_multi)
    models['knn'] = knn
    
    # 4. Naive Bayes (Lightweight)
    logger.info("Training Naive Bayes...")
    nb = GaussianNB()
    nb.fit(X_train, y_train_multi)
    models['naive_bayes'] = nb
    
    # 5. K-Means (Clustering)
    logger.info("Training K-Means...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    models['kmeans'] = kmeans
    
    # Calculate Silhouette Score for K-Means (Unsupervised Metric)
    sil_score = silhouette_score(X_scaled, kmeans_labels)
    results['K-Means (Silh.)'] = sil_score
    
    # Save models and scaler
    os.makedirs('models', exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f'models/{name}.joblib')
    joblib.dump(fe.scaler, 'models/scaler.joblib')
    
    logger.info("Models and scaler saved to models/")
    
    # Evaluation
    evaluation_results = []
    evaluation_results.append("=== Advanced Model Evaluation Report ===\n")
    evaluation_results.append(f"K-Means Silhouette Score: {sil_score:.4f} (Measures cluster quality)\n")
    evaluation_results.append("-" * 30 + "\n")
    
    def evaluate_model(name, model, X_t, y_t):
        y_pred = model.predict(X_t)
        acc = accuracy_score(y_t, y_pred)
        evaluation_results.append(f"Model: {name.upper()}")
        evaluation_results.append(f"Test Accuracy: {acc:.4f}")
        evaluation_results.append("Classification Report:")
        evaluation_results.append(classification_report(y_t, y_pred))
        evaluation_results.append("-" * 30)
        return acc

    results['Random Forest'] = evaluate_model("Random Forest", rf, X_test, y_test_multi)
    results['SVM'] = evaluate_model("SVM", svm, X_test_bin, y_test_bin)
    results['KNN'] = evaluate_model("KNN", knn, X_test, y_test_multi)
    results['Naive Bayes'] = evaluate_model("Naive Bayes", nb, X_test, y_test_multi)
    
    # Save evaluation to file
    report_path = 'models/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("\n".join(evaluation_results))
    
    logger.info(f"Evaluation report saved to {report_path}")

    # Visualization - Optimized for GUI display (Smaller size)
    plt.figure(figsize=(7, 5))
    names = list(results.keys())
    values = list(results.values())
    
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    bars = plt.bar(names, values, color=colors, width=0.5)
    
    plt.ylim(0, 1.2)
    plt.ylabel('Score (Accuracy / Silhouette)', fontsize=9)
    plt.title('Complete Model Performance Comparison', fontsize=12, fontweight='bold')
    plt.xticks(rotation=20, ha='right', fontsize=9)
    
    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = 'models/model_comparison.png'
    plt.savefig(plot_path, dpi=90)
    logger.info(f"Comparison chart saved to {plot_path}")

if __name__ == "__main__":
    train_and_save_models()

