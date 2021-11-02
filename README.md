# BubbleMint, a ML-driven cryptocurrency trading bot
This program runs on Coinbase Pro, if you would like to use it as an investment tool, be sure to have a Coinbase Pro account and insert all necessary API information in ```authCredentials.py```. This program uses 4 different machine learning algorithms to generate buy/sell signals for a given asset and time interval. Pre-trained model for Bitcoin/USD is available and ready to use.

## How it works
The file ```subroutines.py``` contains all the methods used to calculate the different technical indicators of a given cryptocurrency asset and timeframe. The file ```gen_transform.py``` reduces the dimensionality of the dataset by selecting the top 200 features using the Random Forest algorithm. The reduced dataset is then used to train 4 different machine learning algorithms, K-nearest neighbor, random forest, gaussian naive bayes and gradient boosting classifier. The 4 models' outputs are combined using a weighted average, and the final outputs are used as raw signals. Below shows the raw predictions on the BTC/USD pair.\
\
![raw](https://user-images.githubusercontent.com/86272122/139788759-5549fe69-1c03-4d94-86c8-39582657bd08.png)

## The stepping operation
As shown above, the raw outputs from the ensembled model have too many buy/sell signals in the same reneral area. To combat this, a step operation is introduced, where the first buy signal will not result in the execution of the purchase, but rather trigger a stoploss and take-profit margin to be set. If a new buy signal is received prior to the price breaking either margins, the margin "steps" with the new signal and a new margin is generated. The purchase will only be executed when prices eventually crosses the margins. The same operation is done on sall signals. Blow shows the same predictions after the stepping operations.\
\
![stepped](https://user-images.githubusercontent.com/86272122/139789031-068c1a99-db77-45bb-972f-750db1c31000.png)

## Data labeling
Historic prices are first transformed into chunks of equal sizes, the minimum and maximum for each chunk is considered a buy and sell label respectively. To visualize the profit and percent gains for a large range of chunk sizes, execute ```python general_test.py```. Different assets often require different chunk sizes, the default chunk size is ```320```.

## Installation and usage
*This program requires the libraries ```sklearn```, ```imblearn``` as well as ```cbpro```.\
*To install, type in terminal\
```git clone https://github.com/SnowCheetos/BubbleMint.git .```
*A pre-trained model for BTC/USD is ready for use. To use the model, execute ```python trader.py```, but make sure to have inserted all the API information in ```authCredentials.py``` for Coinbase Pro. If you would like to train the model on a new asset, type ```python train.py``` and enter the asset, or modify the function inputs first.

## Testing
To visualize performances of each model, execute ```python general_test.py```. To visualize the performance of the ensembled model, execute ```python ensembled_test.py```.
