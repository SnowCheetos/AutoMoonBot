import joblib
import numpy as np
import matplotlib.pyplot as plt
from subroutines import load_data, eval_label

asset = input('Asset: ')

KNN = joblib.load('models/KNN/'+asset+'.pkl')
RF = joblib.load('models/RF/'+asset+'.pkl')
NB = joblib.load('models/NB/'+asset+'.pkl')
GB = joblib.load('models/GB/'+asset+'.pkl')

X_test, Y_test, price = load_data(asset, 'data/', side = 'test', return_price = True)
X_train, Y_train, price_train = load_data(asset, 'data/', side = 'train', return_price = True)

gain = []
for i in range(100, 1000):
    gain.append(eval_label(price, i))
gain = np.array(gain)
x = np.linspace(100, 1000, 900)

KNN_proba = KNN.predict(X_test)
RF_proba = RF.predict(X_test)
NB_proba = NB.predict(X_test)
GB_proba = GB.predict(X_test)

B_KNN, S_KNN = np.where(KNN_proba == -1)[0], np.where(KNN_proba == 1)[0]
B_RF, S_RF = np.where(RF_proba == -1)[0], np.where(RF_proba == 1)[0]
B_NB, S_NB = np.where(NB_proba == -1)[0], np.where(NB_proba == 1)[0]
B_GB, S_GB = np.where(GB_proba == -1)[0], np.where(GB_proba == 1)[0]

plt.figure(figsize = (16, 8))
plt.plot(x, gain[:,0])
plt.title('Total gain')
plt.xlabel('Chunk Size')
plt.plot()

plt.figure(figsize = (16, 8))
plt.plot(x, gain[:,1])
plt.title('Average gain')
plt.xlabel('Chunk Size')
plt.show()

plt.figure(figsize = (16, 8))
plt.plot(x, gain[:,2])
plt.title('Total profit')
plt.xlabel('Chunk Size')
plt.plot()

plt.figure(figsize = (16, 8))
plt.plot(x, gain[:,3])
plt.title('Average profit')
plt.xlabel('Chunk Size')
plt.show()

plt.figure(figsize = (16, 8))
plt.plot(price)
plt.plot(B_KNN, price[B_KNN], "^")
plt.plot(S_KNN, price[S_KNN], "v")
plt.title('KNN')
plt.show()

plt.figure(figsize = (16, 8))
plt.plot(price)
plt.plot(B_RF, price[B_RF], "^")
plt.plot(S_RF, price[S_RF], "v")
plt.title('Random Forest')
plt.show()

plt.figure(figsize = (16, 8))
plt.plot(price)
plt.plot(B_NB, price[B_NB], "^")
plt.plot(S_NB, price[S_NB], "v")
plt.title('Naive bayes')
plt.show()

plt.figure(figsize = (16, 8))
plt.plot(price)
plt.plot(B_GB, price[B_GB], "^")
plt.plot(S_GB, price[S_GB], "v")
plt.title('Gradient Boost')
plt.show()


