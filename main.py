import os
import sys
import logging
from gui.app import run_app

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_setup():
    required_models = [
        'models/random_forest.joblib',
        'models/svm.joblib',
        'models/knn.joblib',
        'models/naive_bayes.joblib',
        'models/kmeans.joblib',
        'models/scaler.joblib'
    ]
    
    missing = [m for m in required_models if not os.path.exists(m)]
    
    if missing:
        logger.warning(f"Missing models: {missing}")
        logger.info("Attempting automatic setup (generating data and training models)...")
        
        try:
            from data.generate_dummy_data import generate_meeting_data
            from training.train_models import train_and_save_models
            
            logger.info("Generating dummy data...")
            generate_meeting_data()
            
            logger.info("Training models...")
            train_and_save_models()
            
            logger.info("Setup complete!")
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            print("\nPlease run the following commands manually:")
            print("python data/generate_dummy_data.py")
            print("python training/train_models.py")
            sys.exit(1)

def main():
    # Set PYTHONPATH to include the current directory
    os.environ['PYTHONPATH'] = os.getcwd()
    
    logger.info("Starting Meeting Engagement Analyzer...")
    check_and_setup()
    
    run_app()

if __name__ == "__main__":
    main()

