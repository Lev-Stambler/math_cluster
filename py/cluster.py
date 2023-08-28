from typing import List, Tuple
from langchain import LLMChain, OpenAI, PromptTemplate
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


def relative_labels(thms1: List[str], thms2: List[str], centroid_idx1: int, centroid_idx2: int, api_key=None) -> str:
    llm = OpenAI(model_name="gpt-4", openai_api_key=api_key)
    template = """Given the following two labeled sets of Lean theorems, can you describe the main difference in one sentence?

Set {label1}: "{set1}"

Set {label2}: "{set2}"
"""
    prompt = PromptTemplate(template=template, input_variables=["set1", "set2", "label1", "label2"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.run(set1="\n".join(thms1), set2="\n".join(thms2), label1=f"Cluster {centroid_idx1}", label2=f"Cluster {centroid_idx2}")
    