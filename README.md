## BubbleMint, a ML-driven cryptocurrency trading bot\
This program runs on Coinbase Pro, if you would like to use it as an investment tool, be sure to insert all necessary API information in ```authCredentials.py```. This program uses 4 different machine learning algorithms to generate buy/sell signals for a given asset and time interval. Pre-trained model for Bitcoin/USD is available and ready to use.

# How it works\
The file ```subroutines.py``` contains all the methods used to calculate the different technical indicators of a given cryptocurrency asset and timeframe. The file ```gen_transform.py``` reduces the dimensionality of the dataset by selecting the top 200 features using the Random Forest algorithm. The reduced dataset is then used to train 4 different machine learning algorithms, K-nearest neighbor, random forest, gaussian naive bayes and gradient boosting classifier. The 4 models' outputs are combined using a weighted average, and the final outputs are used as raw signals. Below shows the raw predictions on the BTC/USD pair.\
![raw](https://user-images.githubusercontent.com/86272122/139788759-5549fe69-1c03-4d94-86c8-39582657bd08.png)

# The stepping operation\
As shown above, the raw outputs from the ensembled model have too many buy/sell signals in the same reneral area. To combat this, a step operation is introduced, where the first buy signal will not result in the execution of the purchase, but rather trigger a stoploss and take-profit margin to be set. If a new buy signal is received prior to the price breaking either margins, the margin "steps" with the new signal and a new margin is generated. The purchase will only be executed when prices eventually crosses the margins. The same operation is done on sall signals. Blow shows the same predictions after the stepping operations.\
![stepped](https://user-images.githubusercontent.com/86272122/139789031-068c1a99-db77-45bb-972f-750db1c31000.png)

# Installation and usage\
To install, type in terminal ```
