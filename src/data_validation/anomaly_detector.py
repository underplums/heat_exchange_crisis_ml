from sklearn.ensemble import IsolationForest


class AnomalyDetector:

    def __init__(self):

        self.model = IsolationForest(
            contamination=0.02
        )


    def fit(self, X):

        self.model.fit(X)


    def predict(self, X):

        return self.model.predict(X)