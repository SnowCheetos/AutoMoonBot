import joblib
import numpy as np
from imblearn.over_sampling import RandomOverSampler

from utils import to_onehot

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
        preds = np.dstack([
            self.KNN.predict_proba(X),
            self.RF.predict_proba(X),
            self.NB.predict_proba(X),
            self.GB.predict_proba(X)
        ])
        preds = np.einsum("ijk,k->ij", preds, self.w)
        return preds.argmax(1)
