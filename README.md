# A Data Driven Cryptocurrency Trading Bot
* ## You will probably not make money with this, you are at your own risk if you choose to put in real money!
* This program is built with simplicity in mind, you can train/test/run a model for a certain asset in less than 30 minutes.
* This program runs on Coinbase Pro, if you would like to use it as an investment tool, be sure to have a Coinbase Pro account and insert all necessary API information in ```authCredentials.py```. 
* This program uses 4 different machine learning algorithms to generate buy and sell signals for a given asset and time interval. 
* A pre-trained model for ```BTC/USD``` is available and ready to be used.

## How it works
* The file ```subroutines.py``` contains all the methods used to calculate the different technical indicators of a given asset.
* The file ```gen_transform.py``` reduces the dimensionality of the dataset by selecting the top 200 features using Random Forest. 
* The reduced dataset is then used to train 4 different machine learning algorithms, K-nearest neighbor classifier, random forest classifier, gaussian naive bayes classifier and gradient boosting classifier.
*  The 4 models' outputs are combined using a weighted average, and the final outputs are used as raw predictions. Below shows the raw predictions on the ```BTC/USD``` pair.\
\
![raw](https://user-images.githubusercontent.com/86272122/139788759-5549fe69-1c03-4d94-86c8-39582657bd08.png)

## Data Labeling
* Historic prices are first transformed into chunks of equal sizes, the minimum and maximum for each chunk is considered a buy and sell label respectively. 
* To visualize the profit and percent gains for a large range of chunk sizes, execute the line ```python general_test.py```
* Different assets often require different chunk sizes, the default chunk size is ```320```.

## Prediction Processing
* The raw outputs from the ensembled model have too many buy/sell signals in the same general area. 
* To combat this, every time a buy signal is received, it won't immediately trigger a buy action, but rather sets up a stop-loss and take-profit margin that centers at the previous closing price.
* The margins are set up according to the risk tolerance and multiplier settings in ```trader.py```.
* If a new buy signal is received before price breaks the margin, then a new margin will be set at the previous closing price.
* The buy action will only be executed when prices eventually crosses either the stop-loss or take-profit. 
* The same operation is done on sell signals. Blow shows the same predictions after the prediction processing.\
\
![reduced](https://user-images.githubusercontent.com/86272122/139963255-fbecb351-fc31-47c1-880b-c6a71423d9ba.png)

## Live Trading
* The file ```trader.py``` contains the real-time prediction processing methods.
* You might see different results when you change the risk tolerance and risk multiplier values.

## Installation and usage
* This program requires the packages ```sklearn```, ```termcolor```, ```imblearn``` as well as ```cbpro```.
* To install, clone this repo via ```git clone https://github.com/SnowCheetos/Emsemble-Tradebot.git```.
* A pre-trained model for ```BTC/USD``` is ready to be used. To use the model, execute the line ```python trader.py``` and type in ```BTC``` for the asset.
* Make sure to have inserted all the API information in ```authCredentials.py``` for Coinbase Pro.
* If you would like to train the model on an asset other than BTC, execute ```python train.py``` and enter the asset. Make sure the asset is available for trading on Coinbase Pro (XRP is not).
* You are encouraged to change the function parameters to what works best.
* You are also encouraged to set up a general stop-loss when you are holding a certain asset. Cryptocurrency prices are extremely volatile and the model can make bad decisions during extreme volatility.

## Testing
* To visualize performances of each model, execute ```python general_test.py```. 
* To visualize the performance of the ensembled model, execute ```python ensembled_test.py```.
