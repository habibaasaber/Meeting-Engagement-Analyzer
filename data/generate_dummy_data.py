import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_meeting_data(n_students=1000, meeting_duration=60):
    np.random.seed(42)
    
    names = [f"Student {i+1}" for i in range(n_students)]
    
    # Meeting starts at 10:00 AM
    meeting_start = datetime(2026, 5, 1, 10, 0, 0)
    
    data = []
    for name in names:
        # Engagement profiles: High, Medium, Low
        profile = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.4, 0.3])
        
        if profile == 'high':
            join_delay = np.random.randint(0, 5)
            duration = np.random.randint(55, 61)
            chat_count = np.random.randint(5, 20)
            mic_count = np.random.randint(3, 10)
            screen_share = np.random.randint(0, 3)
        elif profile == 'medium':
            join_delay = np.random.randint(0, 15)
            duration = np.random.randint(40, 55)
            chat_count = np.random.randint(1, 8)
            mic_count = np.random.randint(0, 4)
            screen_share = np.random.randint(0, 2)
        else: # low
            join_delay = np.random.randint(0, 30)
            duration = np.random.randint(5, 40)
            chat_count = np.random.randint(0, 3)
            mic_count = np.random.randint(0, 2)
            screen_share = 0
            
        join_time = meeting_start + timedelta(minutes=join_delay)
        leave_time = join_time + timedelta(minutes=duration)
        
        data.append({
            'Name': name,
            'Join Time': join_time.strftime('%H:%M:%S'),
            'Leave Time': leave_time.strftime('%H:%M:%S'),
            'Duration (minutes)': duration,
            'Chat Messages Count': chat_count,
            'Microphone Activity': mic_count,
            'Screen Share Count': screen_share
        })
        
    df = pd.DataFrame(data)
    
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    csv_path = os.path.join('data', 'meeting_report.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated dummy data at {csv_path}")

if __name__ == "__main__":
    generate_meeting_data()
