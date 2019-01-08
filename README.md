# Big Data Challenge Shaastra
Big data challenge by HSBC hosted by Shaastra

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
