# A Data Driven Cryptocurrency Trading Bot
![](media/demo.gif)

## [**DISCLAIMER**] Beating the market is very difficult. It's likely that you will not make money with this, you are at your own risk if you choose to put in real money!

## Introduction

Who haven't though about using ML for stock market predictions! This repository offers educational content for how a trade bot can be built in python using relatively simple components. 

For this purpose, 3 relatively common classifiers were chosen: `K-Nearest-Neighbor`, `Gaussian Naive-Bayes` and `Random Forest` from the `scikit-learn` library. Inputs to the model consists of normalized technical indicator values ranging from (0, 1), and the three models will classify the input to one of three classes: **-1** or a **buy** signal, **+1** or a **sell** signal, **0** or a **hold** signal.

The model is trained live, every new 50 data points. This number is configurable. The targets used for training are produced by first splitting the market price data into equal chunks of size `n`, and then the minimum index of each chunk is recorded and labeled `sell`, the maximum index of each chunk and recorded and labeled `buy`, everything else is labeled `hold`.

After each training session, each model will make a prediction on some recent testing data, and a weight will be assigned to each model that's proportional to the test accuracy values. The predictions made by each model will be linearly combined using their corresponding weight values to produce the final prediction.

## How it works
* The file ```utils.py``` contains all the methods used to calculate the different technical indicators of a given asset.
* The file ```gen_transform.py``` reduces the dimensionality of the dataset by selecting the top 200 features using Random Forest. 
* The reduced dataset is then used to train 4 different machine learning algorithms, K-nearest neighbor classifier, random forest classifier, gaussian naive bayes classifier and gradient boosting classifier.
*  The 4 models' outputs are combined using a weighted average, and the final outputs are used as raw predictions. Below shows the raw predictions on the ```BTC/USD``` pair.\
\
![raw](https://user-images.githubusercontent.com/86272122/139788759-5549fe69-1c03-4d94-86c8-39582657bd08.png)

## Prediction Processing
* The raw outputs from the ensembled model have too many buy/sell signals in the same general area. 
* To combat this, every time a buy signal is received, it won't immediately trigger a buy action, but rather sets up a stop-loss and take-profit margin that centers at the previous closing price.
* The margins are set up according to the risk tolerance and multiplier settings in ```trader.py```.
* If a new buy signal is received before price breaks the margin, then a new margin will be set at the previous closing price.
* The buy action will only be executed when prices eventually crosses either the stop-loss or take-profit. 
* The same operation is done on sell signals. Blow shows the same predictions after the prediction processing.\
\
![reduced](https://user-images.githubusercontent.com/86272122/139963255-fbecb351-fc31-47c1-880b-c6a71423d9ba.png)
