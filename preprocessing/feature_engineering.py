import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def preprocess(self, df):
        # Handle column names (make them consistent)
        df.columns = [col.strip() for col in df.columns]
        
        # 1. duration_ratio = duration / total_meeting_time
        # Assuming total meeting time is the max duration in the report if not provided
        total_time = df['Duration (minutes)'].max()
        df['duration_ratio'] = df['Duration (minutes)'] / total_time if total_time > 0 else 0
        
        # 2. participation_score = chat_count + mic_count + screen_share
        df['participation_score'] = (
            df['Chat Messages Count'] + 
            df['Microphone Activity'] + 
            df.get('Screen Share Count', 0)
        )
        
        # 3. late_join = 1 if join_time > meeting_start else 0
        # For simplicity, we define late as joining more than 5 minutes after the earliest joiner
        earliest_join = pd.to_datetime(df['Join Time']).min()
        df['late_join'] = (pd.to_datetime(df['Join Time']) > earliest_join + pd.Timedelta(minutes=5)).astype(int)
        
        # Feature vector for ML
        features = ['duration_ratio', 'participation_score', 'late_join']
        X = df[features].copy()
        
        return X

    def scale_features(self, X, fit=False):
        if fit:
            self.scaler.fit(X)
            self.is_fitted = True
        
        if self.is_fitted:
            return self.scaler.transform(X)
        else:
            return X # Return unscaled if not fitted yet

def get_labels(df):
    """
    Helper to generate labels for dummy data training.
    In a real scenario, these would be provided or derived from ground truth.
    """
    # Simple logic for synthetic labels
    labels = []
    binary_labels = []
    
    for _, row in df.iterrows():
        score = row['duration_ratio'] * 0.5 + (row['participation_score'] / 20) * 0.5
        
        if score > 0.7:
            labels.append('High')
            binary_labels.append('Engaged')
        elif score > 0.3:
            labels.append('Medium')
            binary_labels.append('Engaged')
        else:
            labels.append('Low')
            binary_labels.append('Disengaged')
            
    return np.array(labels), np.array(binary_labels)
