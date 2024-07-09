# An Ensemble of Models Trading Together

*Please review the disclaimers [`here`](../../DISCLAIMER.MD) before proceeding. By continue visiting the rest of the content, you agree to all the terms and conditions of the disclaimers.*

## Introduction

*Please note that this is a documentation for legacy code, which as of July 2024 is no longer being actively maintained or supported. Although a lot of insight can surely be gained from it, features are not guaranteed to function properly, use for research and educational purposes only. For more up-to-date developments, check out the current version.*

### Demonstration
![](media/demo.gif)

Demonstration of system back-testing. Data used for this test was *BTC/USD* with *1 hour* interval

### Motivation and Approach

Who haven't though about using machine learning for stock market predictions 🤑 This repository offers educational content for how a trade bot can be built using relatively simple components, that can match up pretty well against the market. 

The first step to keep it simple is to stay away from deep learning, and rely solely on classical methods, it seems like given the saga created by LLMs like *Chat* have caused many to forget about their existence. Even in this day and age where *transformers* reign supreme, classical machine learning and statistical methods still prove itself useful in many cases, in particular

- Methods such as *Naive Bayes* and *KNN* use a fraction of the compute resources when compared to even relatively light neural networks.

- Classical methods often out-perform neural networks on the same task, particularily when the dataset size is small, and methods such as *Random Forest* are known for its robust ability to generalize.

- Methods such as *Naive Bayes* and *Gaussian Processes* are based on very strong and well understood mathematics, making it much easier interpret and potentially optimize. Adding on to that, methods such as *Naive Bayes* require little to no parameter tuning.

Unfortunately, no single model with no amount of data is going to be able to consistently reject outliers and beat the market 😞 Therefore, this project aims to solve that by utilizing multiple classical ML methods to create an ensemble. Furthermore, the output signal is post processed prior to actions to enhance the quality of the predictions. In addition, to adapt to the highly dynamic nature of financial markets, the models will be actively trained. 

## Intuition

### Model Selections

Theoretically, any number of any models can be used for this task, and models such as Random Forest are ensembles on their own. However, there are a few criteria for model selection that makes some more suitable than others.

- **Classification vs Regression**

    At first it might be tempting to treat this as a regression problem and try to predict the price directly using non-linear regression such as *Gaussian Regression* and *Kernel methods*, or even *autoregressive* methods such as *RNNs* 🤢
    
    Although there are some merits in them, financial markets, albeit sequential and continuous, does not necessarily mean that the stock market is strictly, or even remotely a *function of time*. Rather, it can be much better described as a *stochastic process*, where the current step carries a probability distribution that describes the potential next future step, and each step only depends on the previous step.

    Therefore, rather than predicting the prices, one should aim to predict the probability distribution that can be used to gain insight on whether or not it's a good time to buy or sell, *a.k.a.* classification.
    
    Don't just take my word for it 🙃 [*Louis Bachelier*](https://en.wikipedia.org/wiki/Louis_Bachelier) was credited as the first person to model the stock market as a stochastic process back in 1900. He then went on to create the [*Bachelier model*](https://en.wikipedia.org/wiki/Bachelier_model), which was one of the first models that can be used to determine the fair price of options, and went on to inspire the creation of the famous [*Black-Scholes model*](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model), which for better or worse, was essential to the trillion dollar n̶a̶t̶i̶o̶n̶a̶l̶ d̶e̶b̶t̶ derivatives market today.


- **Output Probability Distribution**

    Like mentioned, it is desirable for the model to output probability distributions rather than deterministic predictions. This criteria puts deterministic models like Support Vector Machine (*SVM*) and *Decision Tree* out of the scope. *Naive Bayes* comes up as a natural choice, it "naively" follows the [*Bayes Theorem*](https://en.wikipedia.org/wiki/Bayes%27_theorem)

    $$P(A \mid B) = { {P(B \mid A) \cdot P(A)} \over P(B)}$$

    or, *the probability of event A occuring that B occured is the product of probability of event B occuring given even A and the probability of event A, divided by the probability of event B*
    
    Intuitively, this makes sense, *e.g.* let's say that we have a dataset that includes the price of an asset over one year, and records of a series of winning trades. If we can somehow quantize the prices into events, we will be able to use Bayes Theorem to compute the probability of buy, sell, and hold distribution for each given moment. For simplicity's sake, we'll use *RSI-14* as that descriptior, and let's quantize the market into three events
    
    - **A:** *RSI-14 < 20*
    - **B:** *RSI-14 > 80*
    - **C:** *20 < RSI-14 < 80*

    similarily we can quantize the actions into three categories

    - 🦧 Buy
    - 🧻 Sell
    - 💎 HODL

    then at any given time $t$, if the *RSI-14* falls within the **C** category, then the probability that $t$ presents a good buying opportunity can be computed as

    $$P(🦧 \mid C) = {P(C \mid 🦧) \cdot P(🦧) \over P(C)}$$

    the probability that $t$ presents a good selling opportunity can be computed as

    $$P(🧻 \mid C) = {P(C \mid 🧻) \cdot P(🧻) \over P(C)}$$

    these expressions can be generalized to as follows

    $$P(a_i \mid RSI_{14}(p_t)) = {P(RSI_{14}(p_t) \mid a_i) \cdot P(a_i) \over P(RSI_{14}(p_t))}$$

    where $a_i$ represents a specific action, and $p_t$ represents the price of the asset at time $t$


- **Light Weight**

    ...

- **Variety is Key**

    ...

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
