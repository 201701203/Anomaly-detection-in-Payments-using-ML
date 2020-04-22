"""
     Group 18 Local Outlier Factor
"""
import numpy as np
from sklearn.neighbors import KDTree

def LocalOutlierFactor(X, k, outlier_threshold = 1.2):

    # KDtree is a space-partitioning data structure for organizing points in a k-dimensional space. 
    # X = dataset, leaf_size = Number of points at which to switch to brute-force. , p = 2 means a euclidean distance metric
    BT = KDTree(X, leaf_size=k, p=2)

    # 
    distance, index = BT.query(X, k)
    distance, index = distance[:, 1:], index[:, 1:] 
    radius = distance[:, -1]

    #Calculate Local Reachability Distance.
    LRD = np.mean(np.maximum(distance, radius[index]), axis=1)
    r = 1. / np.array(LRD)

    #Calculate outlier score.
    outlier_score = np.sum(r[index], axis=1) / np.array(r, dtype=np.float16)
    outlier_score *= 1. / k

    outliers = []

    for i, score in enumerate(outlier_score):
        if score > outlier_threshold:
            outliers.append([X[i], score])

    return outliers
