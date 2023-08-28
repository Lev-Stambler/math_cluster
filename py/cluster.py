from typing import List, Tuple
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# A list of tuples, where the first element is the label and the second is the embedding
Embeddings = List[Tuple[str, List[float]]]

def cluster(embeddings: Embeddings, n_clusters: int, dimensions=2):
    """
    Cluster the embeddings using k-means clustering.
    :param embeddings: A list of tuples, where the first element is the label and the second is the embedding
    """
    #Load Data
    # data = load_digits().data
    data = np.array([e[1] for e in embeddings])
    pca = PCA(dimensions)
    
    #Transform the data
    df = pca.fit_transform(data)
    
    #Import KMeans module
    from sklearn.cluster import KMeans
    
    #Initialize the class object
    kmeans = KMeans(n_clusters=n_clusters)
    
    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
    
    #Getting unique labels
    u_labels = np.unique(label)
    return df, label, u_labels

