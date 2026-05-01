import joblib
import pandas as pd
import numpy as np
import os
import logging
from preprocessing.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from clustering.kmeans import ClusterAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, models_dir='models'):
        try:
            self.rf = joblib.load(os.path.join(models_dir, 'random_forest.joblib'))
            self.svm = joblib.load(os.path.join(models_dir, 'svm.joblib'))
            self.knn = joblib.load(os.path.join(models_dir, 'knn.joblib'))
            self.nb = joblib.load(os.path.join(models_dir, 'naive_bayes.joblib'))
            self.kmeans_model = joblib.load(os.path.join(models_dir, 'kmeans.joblib'))
            self.scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
            
            self.fe = FeatureEngineer()
            self.fe.scaler = self.scaler
            self.fe.is_fitted = True
            
            self.cluster_analyzer = ClusterAnalyzer()
            self.cluster_analyzer.kmeans = self.kmeans_model
            
        except Exception as e:
            logger.error(f"Failed to load models from {models_dir}: {e}")
            raise

    def predict_all(self, df):
        """
        Runs the full prediction pipeline on a meeting report.
        """
        try:
            # 1. Feature engineering
            X = self.fe.preprocess(df)
            
            # Ensure consistent order
            X = X[FEATURE_COLUMNS]
            X_scaled = self.fe.scale_features(X)
            
            # 2. Supervised predictions
            rf_preds = self.rf.predict(X_scaled)
            svm_preds = self.svm.predict(X_scaled)
            nb_preds = self.nb.predict(X_scaled)
            
            # 3. Dynamic Clustering
            # We pass df because it now contains 'engagement_score' after preprocess()
            clusters = self.cluster_analyzer.get_dynamic_labels(X_scaled, df)
            
            # 4. Combine results
            results = df.copy()
            results['Engagement_Level'] = rf_preds
            results['Binary_Engagement'] = svm_preds
            results['Fast_Score_Label'] = nb_preds
            results['Participation_Cluster'] = clusters
            
            # Save results to file as requested
            results.to_csv('output_results.csv', index=False)
            logger.info("Predictions completed and saved to output_results.csv")
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    def get_similar_students(self, df, student_name, k=5):
        """
        Finds the top K similar students using KNN.
        """
        try:
            # Re-preprocess to get scaled features in correct order
            X = self.fe.preprocess(df)
            X = X[FEATURE_COLUMNS]
            X_scaled = self.fe.scale_features(X)
            
            student_idx = df[df['Name'] == student_name].index
            if len(student_idx) == 0:
                return []
            
            idx = student_idx[0]
            distances, indices = self.knn.kneighbors([X_scaled[idx]], n_neighbors=min(k+1, len(df)))
            
            # Return details of similar students (excluding themselves)
            similar_students = []
            for i in indices[0]:
                if i != idx:
                    row = df.iloc[i]
                    similar_students.append({
                        'Name': row['Name'],
                        'Engagement': row.get('Engagement_Level', 'N/A'),
                        'Score': round(row.get('engagement_score', 0), 2)
                    })
            
            return similar_students
            
        except Exception as e:
            logger.error(f"Error finding similar students: {e}")
            return []

