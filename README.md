This project aims to model the behaviour of the Supreme Court of the United States (SCOTUS). Specifically, we present models which can predict the votes given by justices of the SC. The models presented are general and do not restrict to predictions for a particular term or natural court.

Please refer to the [report](https://github.com/mbharad/Scotus_LSTM/blob/master/SCOTUS_Report.pdf) for complete details.

# Notebooks

  - [Data Exploration](https://github.com/mbharad/Scotus_LSTM/blob/master/nbs/Data%20Exploration.ipynb) : This notebook does a prelimnary exploration of the features/data given as input to the model.
  
  - [SCOTUS - Sequence Modeling](https://github.com/mbharad/Scotus_LSTM/blob/master/nbs/SCOTUS%20-%20Sequence%20Modeling.ipynb) : This notebook converts each feature to one hot encoding and then feeds it to an LSTM. We are able to achieve test accuracies of 82% using this model.
  
  - [SCOTUS - LSTM and Embeddings](https://github.com/mbharad/Scotus_LSTM/blob/master/nbs/SCOTUS%20-%20LSTM%20and%20Embeddings.ipynb) : This notebook converts each input feature into an embedding which are then concatenated and passed on to an LSTM layer. We are able to achieve test accuracies of 90% using this model.
