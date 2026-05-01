import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Consistent feature order
FEATURE_COLUMNS = ['duration_ratio', 'participation_score', 'late_join']

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def preprocess(self, df):
        """
        Processes raw meeting report data into numerical features for ML.
        """
        try:
            # Handle column names (make them consistent)
            df.columns = [col.strip() for col in df.columns]
            
            # 1. duration_ratio = student_duration / total_meeting_duration
            # Actual meeting duration = latest Leave Time - earliest Join Time
            join_times = pd.to_datetime(df['Join Time'])
            leave_times = pd.to_datetime(df['Leave Time'])
            
            meeting_start = join_times.min()
            meeting_end = leave_times.max()
            total_duration_mins = (meeting_end - meeting_start).total_seconds() / 60
            
            if total_duration_mins <= 0:
                logger.warning("Total meeting duration is zero or negative. Using max duration instead.")
                total_duration_mins = df.get('Duration (minutes)', pd.Series([0])).max()

            # Student duration from Join/Leave times
            student_durations = (leave_times - join_times).dt.total_seconds() / 60
            df['duration_ratio'] = (student_durations / total_duration_mins).clip(0, 1)
            
            # 2. participation_score = (chat * 1) + (mic * 2) + (screen * 3)
            # Use .get() for robust column handling
            chat = df.get('Chat Messages Count', 0)
            mic = df.get('Microphone Activity', 0)
            screen = df.get('Screen Share Count', 0)
            
            df['participation_score'] = (chat * 1) + (mic * 2) + (screen * 3)
            
            # 3. late_join = 1 if join_time > meeting_start + 5 mins
            df['late_join'] = (join_times > meeting_start + pd.Timedelta(minutes=5)).astype(int)
            
            # 4. Engagement Score (0-100)
            # Normalize participation score (assuming max around 30 for normalization)
            norm_participation = (df['participation_score'] / 30).clip(0, 1)
            df['engagement_score'] = (df['duration_ratio'] * 50) + (norm_participation * 50)
            
            # 5. Engagement Index Feature
            df['engagement_index'] = (df['duration_ratio'] * 0.6) + (norm_participation * 0.4)
            
            # Feature vector for ML (ensure consistent order)
            X = df[FEATURE_COLUMNS].copy()
            
            return X
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def scale_features(self, X, fit=False):
        if fit:
            self.scaler.fit(X)
            self.is_fitted = True
        
        if self.is_fitted:
            return self.scaler.transform(X)
        else:
            logger.warning("Scaler not fitted. Returning unscaled features.")
            return X

def get_labels(df):
    """
    Helper to generate labels for dummy data training.
    """
    labels = []
    binary_labels = []
    
    # Use the calculated engagement_score for labeling
    for _, row in df.iterrows():
        score = row['engagement_score']
        
        if score > 70:
            labels.append('High')
            binary_labels.append('Engaged')
        elif score > 30:
            labels.append('Medium')
            binary_labels.append('Engaged')
        else:
            labels.append('Low')
            binary_labels.append('Disengaged')
            
    return np.array(labels), np.array(binary_labels)

