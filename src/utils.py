from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

def get_top_words_per_cluster(model, vectorizer, n_words=10):
    feature_names = vectorizer.get_feature_names_out()
    
    for cluster_num, center in enumerate(model.cluster_centers_):
        sorted_indices = center.argsort()[-n_words:][::-1]  # Order by relevance
        top_words = feature_names[sorted_indices]
        
        print(f"Cluster {cluster_num}: {', '.join(top_words)}")
