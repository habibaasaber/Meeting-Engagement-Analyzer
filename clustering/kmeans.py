import joblib
import os

class ClusterAnalyzer:
    def __init__(self, model_path='models/kmeans.joblib'):
        if os.path.exists(model_path):
            self.kmeans = joblib.load(model_path)
        else:
            self.kmeans = None

    def get_cluster_name(self, features_scaled):
        if self.kmeans is None:
            return "Model not found"
        
        cluster_id = self.kmeans.predict(features_scaled)[0]
        # Mapping clusters to participation styles
        # This mapping depends on how the clusters were formed during training
        cluster_map = {
            0: "Passive",
            1: "Active", 
            2: "Distracted"
        }
        return cluster_map.get(cluster_id, f"Cluster {cluster_id}")
