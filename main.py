import os
import sys
from gui.app import run_app

def check_models():
    required_models = [
        'models/random_forest.joblib',
        'models/svm.joblib',
        'models/knn.joblib',
        'models/naive_bayes.joblib',
        'models/kmeans.joblib',
        'models/scaler.joblib'
    ]
    missing = [m for m in required_models if not os.path.exists(m)]
    return missing

def main():
    # Set PYTHONPATH to include the current directory
    os.environ['PYTHONPATH'] = os.getcwd()
    
    missing = check_models()
    if missing:
        print("Warning: Some models are missing. Please run training/train_models.py first.")
        print(f"Missing: {missing}")
        # Optionally we could trigger training here
    
    print("Launching Meeting Engagement Analyzer...")
    run_app()

if __name__ == "__main__":
    main()
