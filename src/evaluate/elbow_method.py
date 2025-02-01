from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

def elbow_method(X, max_clusters=10):
    """
    Apply elbow method to determine the ideal number of clusters.
    """
    sse = []  # List of Sum of Squared Errors (SSE)
    
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    # Calculating the "elbow" automatically (inflexion point)
    diffs = np.diff(sse)
    diff_diffs = np.diff(diffs)
    
    best_k = np.argmin(diff_diffs) + 3
    print(f'the best_k found by Elbow Method is: {best_k}')
    
    return best_k