# ML-Project-2

## How to run the code

In order to build the final prediction of the code you will need to run the `run.py` file with 
```
 $ python run.py
```
this will load the premade predictions and use xgboost classifier to make a prediction with the first 5 models.

If you want to run the code from A to Z, you will need the `http://nlp.stanford.edu/projects/glove/` GloVe for twitter data and extract it in data folder. 
Once this is done, you can run the 
```
 $ python model_NN.py
```
file that will make all the predictions. We advise to run the models one by one due to the time it takes to train them.

## Files & Folders

- The root of the project contains the `run.py` file that is used to make the best prediction that we have achieved.

- `models_NN.py` is the file to run in order to clean the tweets and do all the preprocessing, it then produces pickled files that contain all the train and test tweets, 
the labels for the training data. For the first model, we also pickled the embedding matrix so that we can easily import it later for the first 5 neural nets.

- `dict` contains dictionaries for the common typing mistakes.

- `results` contains the predicted probabilities of the problems used for boosting as well as the prediction that are made.

- `data` contains all the twitter data as well as GloVe200 pretrained matrix and the pickled data created by the code.

- the `src` folder contains more code that is called from the root:

- `implementation.py` and `submission.py` contains some useful function that were used throughout the rest of the project.

- `preprocess.py` is used for all the preprocessing of the tweet and rewrites the cleaned tweets to a seperate file. This part of the code is modulable to try different types
of processing.

- `train_test_tokenizer.py` this file does all the heavy computing before the training of the different models and pickles the result of the train/test tweets, the labels attached 
to them as well as the embedding matrix in the cases where we use it. It can also be used to produce the data with n-grams when needed.

we used `wor2vec.py` to create our own Word2Vec embedding of the data then the data it is simply fitted with a logistic regression to compare the accuracy to the other models

`word2vec_ngram.py` is an improved version of the Word2Vec model that can be used with n-grams as well. Again; we fit with a logistic regression to compare to the performance 
of other models.

## Authors

- Emile Bourban
- Thimothée Bronner
- Florian Delberghe