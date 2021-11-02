import joblib
import numpy as np
from imblearn.over_sampling import RandomOverSampler

from subroutines import to_onehot

class ensembled:
    def __init__(self, asset):
        self.asset = asset
        try:
            self.RF = joblib.load('models/RF/'+asset+'.pkl')
            self.KNN = joblib.load('models/KNN/'+asset+'.pkl')
            self.NB = joblib.load('models/NB/'+asset+'.pkl')
            self.GB = joblib.load('models/GB/'+asset+'.pkl')
        except:
            raise Exception("One or few of the models are unavailable!")
        self.w = np.array([0.28, 0.27, 0.1, 0.35])
        self.sampler = RandomOverSampler()

    def predict(self, X):
        self.preds = np.array([
            self.KNN.predict(X),
            self.RF.predict(X),
            self.NB.predict(X),
            self.GB.predict(X)
        ]).T
        self.pred = (self.preds @ self.w).T
        return self.pred

