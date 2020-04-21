"""
     Group 18 Local Outlier Factor
     https://github.com/harisek36/LOF-Anonomly-Detect
"""
import numpy as np
import time
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

"""Generate random clusters."""
def random_clusters(num_tests, num_outliers, dim, outliers = False):
	""" Generate cluster like data set."""
	X = 0.3 * np.random.randn(num_tests//2, dim)
	X_outliers = np.random.uniform(\
		low=-4, high=4, size=(num_outliers, dim))
	X = np.r_[X + 2, X - 2, X_outliers]

	if outliers:
		return X, X_outliers
	else: 
		return X

"""Generate multivariate normal data."""
def multivariate_normal(num_tests, mean, cov):
	X = np.random.multivariate_normal(mean, cov, num_tests)
	return X

def lof(X, k, outlier_threshold = 1.5, verbose = False):

    BT = KDTree(X, leaf_size=k, p=2)

    distance, index = BT.query(X, k)
    distance, index = distance[:, 1:], index[:, 1:] 
    radius = distance[:, -1]

    """Calculate LRD."""
    LRD = np.mean(np.maximum(distance, radius[index]), axis=1)
    r = 1. / np.array(LRD)

    """Calculate outlier score."""
    outlier_score = np.sum(r[index], axis=1) / np.array(r, dtype=np.float16)
    outlier_score *= 1. / k

    if verbose: print ("Recording all outliers with outlier score greater than %s."\
     % (outlier_threshold))

    outliers = []
    """ Could parallelize this for loop, but really not worth the overhead...
        Would get insignificant performance gain."""
    for i, score in enumerate(outlier_score):
        if score > outlier_threshold:
            outliers.append([X[i], score])

    if verbose:
        print ("Detected outliers:")
        print (outliers)

    return outliers

def data_visualization(X,X_outliers):
    """Plot data nicely."""
    plt.scatter(X[:,0], X[:,1], c='yellow')

    all_outliers = []
    scores = []
    for i, pair in enumerate(X_outliers):
        all_outliers.append(pair[0])
        scores.append(pair[1])

    X_o = np.vstack(all_outliers)
    
    plt.scatter(X_o[:,0], X_o[:,1], c='red')

    plt.show()

"""Set K nearest neighbors to look at.
k = 10

data_dim = 2
num_tests = 10000
num_outliers = 2

mean = [1,1]
cov = [[0.3, 0.2],[0.2, 0.2]]

X = random_clusters(num_tests,num_outliers,data_dim)
    #X = data_styles.multivariate_normal(num_tests,mean,cov)

start = time.time()
predicted_outliers = lof(X, k, outlier_threshold = 1.75)

print ("---------------------")
print ("Finding outliers in %s values took %s seconds." % (len(X),time.time() - start))
print ("---------------------")

if data_dim == 2:
    data_visualization(X, predicted_outliers)
"""

