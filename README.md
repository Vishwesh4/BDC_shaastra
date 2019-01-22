# Big Data Challenge Shaastra
Big data challenge by HSBC hosted by Shaastra. The problem statement being, to predict price after 30 seconds in a
limit order book.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Libraries needed

```
re
time
datetime
operator
numpy
pandas
collections
unicodedata
collections
seaborn
matplotlib
tqdm
IPython.display.Image
statsmodels
sklearn
keras
Tensorflow
scipy
```
## Approach
In general, a set of fully connected layers are attached and used to classify the input data. However, all
inputs to the fully connected layers are assumed independent of each other, while our features still contain
time dependencies. In order to capture temporal relationship that exist in the extracted features, we replace
the fully connected layers with LSTM units. The activation of a LSTM unit is fed back to itself and the
memory of past activations is kept with a separate set of weights, so the temporal dynamics of our features
can be modelled. We use 3 layers LSTM in our work, and find using more units can lead to overfitting.
In our model we predict the stock price 30 seconds ahead by actually finding the price movement happening
after 30 secs. We labelled it into 3 classes.
  1) +1 representing tick up
  2) 0 representing no significant change
  3) -1 representing tick down
The last output layer uses a softmax activation function to classify into 3 classes and hence the final output
elements represent the probability of each price movement class (tick up, tick down or no change) at each
time step.

For prediction of predMid, we used:

![Alttext](https://raw.github.com/Vishwesh4/BDC_Shaastra/master/images/equation_image.png)

## Labelling

Because financial data is highly stochastic, if we simply compare pt and pt+k to decide the price movement,
the resulting label set will be noisy. We adopt the idea of introducing a smoothed labelling method. First, m−
denotes the mean of the previous k mid-prices and m+ denotes the mean of the next k mid-prices:

![Alttext](https://raw.github.com/Vishwesh4/BDC_Shaastra/master/images/labelling.png)

where pt is the mid-price defined in Equation (1) and k is the prediction horizon (k = 300 signifying 30
seconds on average). Then, we use m− and m+ to define the direction of price movement (lt) at time t by:
![Alttext](https://raw.github.com/Vishwesh4/BDC_Shaastra/master/images/labelling_2.png)

where the threshold α determines the smallest change in price that must occur for the mid-price to be
considered upward (+1) or downward (−1). We choose the threshold α such that all the classes are
balanced for each instrument during the training period. This can be understood as adjusting for the average
volatility of a given instrument, i.e. if a stock is more volatile we require a larger upwards (downwards) price
move to classify it as a +1 (−1).

## Feature Description

![Alttext](https://raw.github.com/Vishwesh4/BDC_Shaastra/master/images/feature.png)


## Running the tests

You can load data from :-
* `1 - intraday/quote_in.csv` - data for quotes, training set
* `2 - intraday/quote_out.csv` - data for quotes, test set
* `3 - intraday/trade_in.csv` - trade data , training set
* `4 - intraday/trade_out.csv` - trade data , test set

We have some important files
* `1 - model.ipynb` - Main pipeline
* `2 - core/model.py` - contains class model which is our LSTM model
* `3 - core/data_processor.py` - data_loader class for ease of loading big data
* `4 - config.json` - contains all the important parameters for model
* `5 - saved_models` - contains the saved checkpoints of both prediction and model weights
* `6 - model.h5` - contains the current model weights we are using for replicability
