
# Cross Industry Standard Process for Data Mining (CRISP-DM) Methodology for Algo Trading.

## 1. Business Understanding

An algorithmic trading business earns its profits from maximizing the profits of its investments, for a specific risk level. This requires the best estimations of trend and volatility of its assets in multiple timeframes, these estimations are feed to its automatic portfolio balancer based on Modern Portfolio Theory.

## 2. Data Understanding

The data that is most useful for the decision-making process for the business is the trend and volatility of each asset in a portfolio, these data can be influenced by many factors, such as, the current state of the market given by technical indicators, fundamental indicators like fincancial calendar events, and geopolitical events.

Since the methodologies for searching the best predictors for different assets will be probably similar, in this project we will describe the process for estimating trend and volatility of EUR/USD specifically, so future work can replicate the same methodology for other assets.

The data available for using is:
- The historical EUR/USD exchange rates
- Existing dataset of the results (also expected values) of financial events.

## 3. Data Preparation

### Timeseries Data

- Split dataset between training and validation 
- Calculate volatility timeseries
- Unbiasing (substract each tick by the average of the previous n ticks)
- Normalizing (min-max, zero centered?)
- (optional) Calculate Technical Indicators

### Calendar Event Data 

- One-hot encoding 
- CNN-generated from training with unbiased/normalized timeseries

### Feature Selection

Use statistical methods (correlation, mutual-information and others) to remove redundant features or features containing little information useful for the prediction.

### Feature Extraction

-Autoencoder with a sliding window technique with the window size equal to encoder input size.
-Transforms a single timeseries(column) into a multi-column dataset with the number of columns equal to the reduced dimension (encoder output size), ready to be feed to a prediction model for training.

### Prediction Training Data

The data to be estimated, and to be used as trainig signal, is the variation in the close price in multiple timeframes (15min, 1h, 4h, 1d, 1w), and also the volatility as the standard deviation of logarithmic returns (in the previous timeframe).

## 4. Modeling

Having all the resulting features of the data preparation, select the most suitable estimation model for the required training signal. Model selection implies the configuration of hyperparameters and training of the models with preprocessed data trying to aproximate the training signal. Models to test: SVM, ANN, CNN, and 
others.

### Error Measurements:

The error both in the exchange rate trend and volatility will be MSE between the training signal and the aproximated signal (output of predictor).

### (optional) Automatic search for feature extractors and predictor configuration
The confiugurationof both feature extractors and predictor models is composed of the structure (number of layers, types of layers, interconnections, modules) and hyperparameters such as the number of filters per layer, the activation fuctions of layers, etc...

Initially this configuration is made manually by setting up a setructure and initial parameters before training the model,but ideally an automatic search of parameters can be made using different techniques such as incremental prunning or genetic algorithms.

### Synthetic data

Synthetic data will be generated with statistical and generative methods to verify during evaluation if its use increments the performance of the predictor.

### 5. Evaluation

The evaluation of the trained predictor is to be made with the data selected for validation, which is to be preprocessed and feed to the predictor to obtain a predicted timeseries which will be compared to the validation training signal to obtain the error ofthe prediction for de fdifferent models and configurations.

The results of the evaluation may trigger adjustments on the model with the objective of improving the performance of the predictor (reduction of the error).

Measurements of error of the prediction with an incremental quantity of training data, are to be made. Also  measurements will be made to see if the addition of real training data with synthetic data, reduces the error.

### 6. Deployment

The deployment of the system can be made by using the trained feature-extractors and predictor in simulations with real data (and synthetic data depending on results). If the simulations are working correctly, the trained models can be incorporated in the bussiness workflow, and contiuous improvement policies must be implemented to keep the models updated.

Also, other assets may use the similar techniques to aproximate their trends and volatilies.







