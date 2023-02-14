import joblib
import numpy as np
from utils import load_data
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier

random_sampler = RandomOverSampler()

KNN = KNeighborsClassifier(
    n_neighbors = int(np.sqrt(200)), 
    n_jobs = -1
)

RF = RandomForestClassifier(
    n_estimators = 10000,
    criterion = 'gini', 
    min_samples_split = 0.01,
    min_samples_leaf = 0.01,
    max_features = 'sqrt', 
    n_jobs = -1, 
    random_state = 2,
    verbose = 9, 
)

NB = GaussianNB()

GB = GradientBoostingClassifier(
    n_estimators = 1000, 
    min_samples_split = 0.05,
    min_samples_leaf = 0.05,
    verbose = 9
)


def model_gen(asset, X, Y):
    X_train, Y_train = random_sampler.fit_resample(X, Y)
    KNN.fit(X_train, Y_train)
    RF.fit(X_train, Y_train)
    joblib.dump(KNN, 'models/KNN/'+asset+'.pkl')
    joblib.dump(RF, 'models/RF/'+asset+'.pkl')

if __name__ == "__main__":
    X, Y = load_data('BTC', 'data/')
    X_train, Y_train = random_sampler.fit_resample(X, Y)
    NB.fit(X_train, Y_train)
    joblib.dump(NB, "models/NB/BTC.pkl")
    GB.fit(X_train, Y_train)
    joblib.dump(GB, "models/GB/BTC.pkl")
