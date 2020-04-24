"""
     Group 18 Support Vector Machine
"""
import numpy as np

 # RBF kernel
def GaussianKernel(v1, v2, sigma):
    return np.exp(-np.subtract(v1-v2, 2)**2/(2.*sigma**2))

class SupportVectorMachine:
    def __init__(self, C = 1, sigma=1, epochs=0):
        self.C = C
        self.sigma = sigma
        self.epochs = epochs

    # Sequential minimal optimization
    def __smo(self, X, y, kernel):
        n_samples = X.shape[0]

        alpha = np.zeros(n_samples)
        self.__bias = 0
        e = -y

        for _ in range(self.epochs):
            for i in range(n_samples):
                hi = kernel[i].dot(alpha * y) + self.__bias
                if (y[i] * hi < 1 and alpha[i] < self.C) or (y[i] * hi > 1 and alpha[i] > 0):
                    j = np.argmax(np.abs(e - e[i]))

                    if y[i] == y[j]:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    else:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    if L == H:
                        continue

                    eta = kernel[i, i] + kernel[j, j] - 2 * kernel[i, j]
                    if eta <= 0:
                        continue

                    alpha_j = alpha[j] + y[j] * (e[i] - e[j]) / eta

                    if alpha_j > H:
                        alpha_j = H
                    elif alpha_j < L:
                        alpha_j = L

                    alpha_i = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j)

                    bi = self.__bias - e[i] - y[i] * kernel[i, i] * (alpha_i - alpha[i]) - y[j] * kernel[i, j] * (alpha_j - alpha[j])
                    bj = self.__bias - e[j] - y[i] * kernel[i, j] * (alpha_i - alpha[i]) - y[j] * kernel[j, j] * (alpha_j - alpha[j])

                    if 0 < alpha_i and alpha_i < self.C:
                        self.__bias = bi
                    elif 0 < alpha_j and alpha_j < self.C:
                        self.__bias = bj
                    else:
                        self.__bias = (bi + bj) / 2

                    alpha[i] = alpha_i
                    alpha[j] = alpha_j

                    e[i] = kernel[i].dot(alpha * y) + self.__bias - y[i]
                    e[j] = kernel[j].dot(alpha * y) + self.__bias - y[j]
                
        support_items = np.flatnonzero(alpha > 1e-6)
        self.__X_support = X[support_items]
        self.__y_support = y[support_items]
        self.__a_support = alpha[support_items]

    def fit(self, X, y):
        kernel = GaussianKernel(X, X, self.sigma)
        self.__smo(X, y, kernel)


    def predict(self, X):
        return np.sign(self.score(X))


    def score(self, X):
        kernel = GaussianKernel(X, self.__X_support, self.sigma)
        return (self.__a_support * self.__y_support).dot(kernel) + self.__bias
