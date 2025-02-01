from sklearn.cluster import KMeans

def clusterize(X, num_clusters):
    """
    Do KMeans clustering.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(X)  # clustering and obtaining os clusters labels
    
    return labels, kmeans
