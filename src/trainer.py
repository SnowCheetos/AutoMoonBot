import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from src.utils import *

class Trainer:
    def __init__(self, num_periods=4):
        # class_weight = 'balanced'
        self.nb = GaussianNB()
        self.rf = RandomForestClassifier(
            n_estimators=321,
            class_weight="balanced",
            n_jobs=4
        )
        self.kn = KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=4
        )
        self.periods = np.logspace(3, num_periods+2, num=num_periods, base=2).astype(int)
        self.col_idx = np.random.permutation(np.arange(5 * self.periods.shape[0]))
        self.confs = np.ones(3) / 3

    def transform(self, x):
        rsis = np.stack([calculate_rsi(x, p) for p in self.periods], axis=-1)
        smas = np.stack([calculate_sma(x, p) for p in self.periods], axis=-1)
        emas = np.stack([calculate_ema(x, p) for p in self.periods], axis=-1)
        blbs = np.stack([calculate_bollinger_bands(x, p) for p in self.periods], axis=-1)
        stos = np.stack([calculate_stochastic_oscillator(x, p) for p in self.periods], axis=-1)
        features = np.hstack([
            rsis, smas, emas, blbs, stos
        ])
        features = features[:, self.col_idx]
        return features

    def fit(self, x, chunk_size=72):
        x = self.transform(x)
        rows_without_nan = ~np.any(np.isnan(x), axis=1)
        indices = np.where(rows_without_nan)[0]
        y = label_data(x, chunk_size)
        x, y = x[indices], y[indices]
        x, y = shuffle(x, y)
        n = int(0.8 * x.shape[0])
        x_train, x_test = x[:n], x[n:]
        y_train, y_test = y[:n], y[n:]
        print("[INFO] Training models... \n")
        self.kn.fit(x_train, y_train)
        print("[INFO] K-Nearest-Neighbor trained. \n")
        self.nb.fit(x_train, y_train)
        print("[INFO] Naive-Bayes trained. \n")
        self.rf.fit(x_train, y_train)
        print("[INFO] Random-Forest trained. \n")
        self.assign_weights(x_test, y_test)
        print("[INFO] Weight assigned, training complete. \n")

    def predict(self, x):
        x = self.transform(x)
        rows_without_nan = ~np.any(np.isnan(x), axis=1)
        indices = np.where(rows_without_nan)[0]
        x = x[indices]
        preds = np.stack([
            self.kn.predict_proba(x),
            self.rf.predict_proba(x),
            self.nb.predict_proba(x)
        ], -1)
        preds = np.einsum("ijk,k->ij", preds, self.confs)
        return preds.argmax(1) - 1

    def assign_weights(self, x, y):
        accuracies = np.asarray([
            accuracy_score(self.nb.predict(x), y),
            accuracy_score(self.kn.predict(x), y),
            accuracy_score(self.rf.predict(x), y)
        ])
        self.confs = accuracies / np.linalg.norm(accuracies)

    def __call__(self, x):
        return self.predict(x)