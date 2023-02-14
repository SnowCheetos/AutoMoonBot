import math
import numpy as np
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

feature_selector = RandomForestClassifier(
    n_estimators = 10000,
    criterion = 'gini', 
    max_depth = None,
    min_samples_split = 0.05,
    min_samples_leaf = 0.05, 
    min_weight_fraction_leaf = 0.0, 
    max_features = 'sqrt', 
    max_leaf_nodes = None, 
    bootstrap = True, 
    oob_score = False,
    n_jobs = -1, 
    random_state = None,
    verbose = 9, 
    warm_start = False, 
    class_weight = None
)

random_sampler = RandomOverSampler()

def get_data_full(asset, chunk_size, start, end = None, interval = '15m', source = 'coinbasepro'):
    output = pd.DataFrame()
    dataset = get_data(asset, start, end, interval, source)
    output['rGradient(data,"close")'] = rGradient(dataset, 'close')
    output['rGradient(data,"open")'] = rGradient(dataset, 'open')
    output['rGradient(data,"high")'] = rGradient(dataset, 'high')
    output['rGradient(data,"low")'] = rGradient(dataset, 'low')
    output['ratio(data,"high","low")'] = ratio(dataset, 'high', 'low')
    output['ratio(data,"open","close")'] = ratio(dataset, 'open', 'close')
    for i in range(4, 40):
        BOL = rBollinger(dataset, 'close', i)
        output['rBollinger(data,"close",' + str(i) + ')[0]'] = BOL[0]
        output['rBollinger(data,"close",' + str(i) + ')[1]'] = BOL[1]
        output['rBollinger(data,"close",' + str(i) + ')[2]'] = BOL[2]
    for i in range(4, 40):
        output['RSI(data,"close",' + str(i) + ')'] = RSI(dataset, 'close', i)
    for i in range(4, 40):
        output['RSI(data,"open",' + str(i) + ')'] = RSI(dataset, 'open', i)
    for i in range(20, 60):
        output['rMACD(data,"close",' + str(i) + ',' + str(math.floor(i/2)) + ',9)'] = rMACD(dataset, 'close', i, math.floor(i/2), 9)
    for i in range(20, 60):
        output['rMACD(data,"open",' + str(i) + ',' + str(math.floor(i/2)) + ',9)'] = rMACD(dataset, 'open', i, math.floor(i/2), 9)
    for i in range(4, 40):
        RATR = rATR(dataset, i)
        output['rATR(data,' + str(i) + ')[0]'] = RATR[0]
        output['rATR(data,' + str(i) + ')[1]'] = RATR[1]
        output['rATR(data,' + str(i) + ')[2]'] = RATR[2]
    for i in range(10, 300):
        output['rEMA(data,"close",' + str(i) + ')'] = rEMA(dataset, 'close', i)
    for i in range(10, 300):
        output['rSMA(data,"close",' + str(i) + ')'] = rSMA(dataset, 'close', i)
    output['Actions'] = labelData(dataset, chunk_size)
    output['Close'] = np.array(dataset.close)
    return output[int(0.1*output.shape[0]):]

def rank_features(asset, data, selector, sampler, download_feat = True, show_fig = False):
    features = data.columns[:-3].tolist()
    X, Y = np.array(data)[:, :-2], np.array(data)[:, -2]
    Xt, Yt = sampler.fit_resample(X, Y)
    selector.fit(Xt, Yt)
    ranks = pd.DataFrame(data = zip(features, selector.feature_importances_), columns = ['feature', 'score'])
    ranks = ranks.sort_values(by = "score", ascending = False)
    if download_feat == True:
        ranks[:50].to_csv('data/features/'+asset+'(50).csv')
        ranks[:100].to_csv('data/features/'+asset+'(100).csv')
        ranks[:200].to_csv('data/features/'+asset+'(200).csv')
        ranks[:400].to_csv('data/features/'+asset+'(400).csv')
        ranks.to_csv('data/features/'+asset+'(all).csv')
    plt.figure(figsize = (16, 8))
    plt.plot(np.array(ranks['score']))
    plt.savefig('data/features/'+asset+'.png')
    if show_fig == True:
        plt.show()
    return ranks

def filter(data, ranks):
    features = ranks['feature'].tolist()
    features.append('Actions')
    features.append('Close')
    return data[features]

def split(data, training_split = 0.8):
    training_len = int(training_split * data.shape[0])
    train, test = data[:training_len], data[training_len:]
    return train, test

def gen_transform(asset, chunk_size, start, end = None, num_top_features = 200):
    dataset = get_data_full(asset, chunk_size, start, end = end)
    ranks = rank_features(asset, dataset, feature_selector, random_sampler)[:num_top_features]
    filtered = filter(dataset, ranks)
    train, test = split(filtered)
    train.to_csv('data/train/'+asset+'.csv', index = False)
    test.to_csv('data/test/'+asset+'.csv', index = False)
    return True

if __name__ == "__main__":
    gen_transform('BTC', 320, 100, 50)