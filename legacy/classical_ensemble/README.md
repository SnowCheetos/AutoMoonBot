# A Data Driven Cryptocurrency Trading Bot

*Please review the disclaimers [`here`](../../README.md) before proceeding. By continue visiting the rest of the content, you agree to all the terms and conditions of the disclaimers.*

## Introduction

*Please note that this is a documentation for (pretty old) legacy code, which as of July 2024 is no longer being actively maintained or supported. Although a lot of insight can surely be gained from it, features are not guaranteed to function properly, use for research and educational purposes only. For more up-to-date developments, check out the current version.*

### Demonstration
![](media/demo.gif)

Demonstration of system back-testing. Data used for this test was *BTC/USD* with *1 hour* interval

### Motivation and Approach

Who haven't though about using machine learning for stock market predictions! This repository offers educational content for how a trade bot can be built using relatively simple components, that can match up pretty well against the market. 

The first step to keep it simple is to stay away from deep learning, and rely solely on classical methods, it seems like given the saga created by *LLMs* have caused many to forget about their existence. Even in this day and age where *transformers* reign supreme, classical machine learning and statistical methods still prove itself useful in many cases, in particular

- Methods such as *Naive Bayes* and *KNN* use a fraction of the compute resources when compared to even relatively light neural networks.

- Classical methods often out-perform neural networks on the same task, particularily when the dataset size is small, and methods such as *Random Forest* are known for its robust ability to generalize.

- Methods such as *Naive Bayes* and *Gaussian Processes* are based on very strong and well understood mathematics, making it much easier interpret and potentially optimize. Adding on to that, methods such as *Naive Bayes* require little to no parameter tuning.

Unfortunately, no single model with no amount of data is going to be able to consistently reject outliers and beat the market. Therefore, this project aims to solve that by utilizing three classical ML methods to create an ensemble. 

Furthermore, the output signal is post processed prior to actions to enhance the quality of the predictions. In addition, to adapt to the highly dynamic nature of financial markets, the models will be actively trained. 

## Intuition

### Ensemble

Theoretically, any number of any models can be used for this task, and models such as Random Forest are ensembles on their own. However, there are a few criteria for model selection that makes some more suitable than others.

- **Classification vs Regression**

    At first it might be tempting to treat this as a regression problem and try to predict the price directly using non-linear regression such as *Gaussian Regression* and *Kernel methods*, or even *autoregressive* methods such as *RNNs*. Although there are some merits in them, financial markets, albeit sequential and continuous, does not necessarily mean that the stock market is strictly, or even remotely a *function of time*. 

- **Output Probability Distribution**

    It is preferrable for the model to output probability distributions as opposed to a deterministic classification. 

- **Light Weight**



- **Variety is Key**

### Data Preprocessing

The model is trained live, every new 50 data points. This number is configurable. The targets used for training are produced by first splitting the market price data into equal chunks of size `n`, and then the minimum index of each chunk is recorded and labeled `sell`, the maximum index of each chunk and recorded and labeled `buy`, everything else is labeled `hold`.

After each training session, each model will make a prediction on some recent testing data, and a weight will be assigned to each model that's proportional to the test accuracy values. The predictions made by each model will be linearly combined using their corresponding weight values to produce the final prediction.

### Post Processing
...

## Usage
* The file ```utils.py``` contains all the methods used to calculate the different technical indicators of a given asset.
* The file ```gen_transform.py``` reduces the dimensionality of the dataset by selecting the top 200 features using Random Forest. 
* The reduced dataset is then used to train 4 different machine learning algorithms, K-nearest neighbor classifier, random forest classifier, gaussian naive bayes classifier and gradient boosting classifier.
*  The 4 models' outputs are combined using a weighted average, and the final outputs are used as raw predictions. Below shows the raw predictions on the ```BTC/USD``` pair.

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
