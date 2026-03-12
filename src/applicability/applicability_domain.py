import numpy as np
from scipy.spatial.distance import mahalanobis


class ApplicabilityDomain:

    def __init__(self):

        self.mean = None
        self.inv_cov = None


    def fit(self, X):

        self.mean = np.mean(X, axis=0)

        cov = np.cov(X, rowvar=False)

        self.inv_cov = np.linalg.pinv(cov)


    def score(self, x):

        return mahalanobis(x, self.mean, self.inv_cov)


    def is_within_domain(self, x, threshold=3.5):

        dist = self.score(x)

        return dist < threshold