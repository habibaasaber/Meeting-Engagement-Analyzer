import joblib
import pandas as pd
import numpy as np
import os
from preprocessing.feature_engineering import FeatureEngineer

class Predictor:
    def __init__(self, models_dir='models'):
        self.rf = joblib.load(os.path.join(models_dir, 'random_forest.joblib'))
        self.svm = joblib.load(os.path.join(models_dir, 'svm.joblib'))
        self.knn = joblib.load(os.path.join(models_dir, 'knn.joblib'))
        self.nb = joblib.load(os.path.join(models_dir, 'naive_bayes.joblib'))
        self.kmeans = joblib.load(os.path.join(models_dir, 'kmeans.joblib'))
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        self.fe = FeatureEngineer()
        self.fe.scaler = self.scaler
        self.fe.is_fitted = True

    def predict_all(self, df):
        # Feature engineering
        X = self.fe.preprocess(df)
        X_scaled = self.fe.scale_features(X)
        
        # Supervised predictions
        rf_preds = self.rf.predict(X_scaled)
        svm_preds = self.svm.predict(X_scaled)
        nb_preds = self.nb.predict(X_scaled)
        
        # Clustering
        cluster_labels = self.kmeans.predict(X_scaled)
        cluster_map = {0: 'Passive', 1: 'Active', 2: 'Distracted'} # Mapping based on training logic
        clusters = [cluster_map.get(c, 'Unknown') for c in cluster_labels]
        
        # Combine results
        results = df.copy()
        results['Engagement_Level'] = rf_preds
        results['Binary_Engagement'] = svm_preds
        results['Fast_Score'] = nb_preds # Naive Bayes
        results['Participation_Cluster'] = clusters
        
        return results

    def get_similar_students(self, df, student_name):
        # Implementation for KNN similarity
        X = self.fe.preprocess(df)
        X_scaled = self.fe.scale_features(X)
        
        try:
            student_idx = df[df['Name'] == student_name].index[0]
        except IndexError:
            return []
            
        distances, indices = self.knn.kneighbors([X_scaled[student_idx]])
        
        # Return names of similar students (excluding themselves)
        similar_names = [df.iloc[i]['Name'] for i in indices[0] if i != student_idx]
        return similar_names
