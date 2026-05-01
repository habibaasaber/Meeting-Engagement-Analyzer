import joblib
import os
import numpy as np
import pandas as pd

class ClusterAnalyzer:
    def __init__(self, model_path='models/kmeans.joblib'):
        if os.path.exists(model_path):
            self.kmeans = joblib.load(model_path)
        else:
            self.kmeans = None

    def get_dynamic_labels(self, X_scaled, df_results):
        """
        Dynamically assigns labels (Active, Passive, Distracted) based on 
        the mean engagement_score of each cluster.
        """
        if self.kmeans is None:
            return ["Unknown"] * len(df_results)
        
        cluster_ids = self.kmeans.predict(X_scaled)
        df_results['cluster_id'] = cluster_ids
        
        # Calculate mean engagement score for each cluster
        cluster_means = df_results.groupby('cluster_id')['engagement_score'].mean().to_dict()
        
        # Sort cluster IDs by mean score (highest = Active, lowest = Distracted)
        sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
        
        mapping = {}
        labels = ["Active", "Passive", "Distracted"]
        
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            if i < len(labels):
                mapping[cluster_id] = labels[i]
            else:
                mapping[cluster_id] = f"Cluster {cluster_id}"
                
        return [mapping.get(cid, "Unknown") for cid in cluster_ids]

