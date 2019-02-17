# -*- coding: utf-8 -*-

import os
import numpy as np
import _pickle as cPickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
import xgboost as xgb

from src import implementations as imp
from src import submisson as sub
from src import preprocess
from src import train_test_tokenizer

def save_model_predict(model, train, test, filename):
    """
    Saves the predictions made on the train and test data, as well as the model for eventual further training
    """
    
    cPickle.dump(model, open("models/{}.pkl".format(filename), 'wb'))
    cPickle.dump(train, open("results/train/train_{}.pkl".format(filename), 'wb'))
    cPickle.dump(test, open("results/test/test_{}.pkl".format(filename), 'wb'))
    
    
def model_1(EMB_DIMS, filepath):
    """
    Using keras, we define the first model with one embedding layer and
    one convolutional layer followed by one maxpooling layer and 2 Dense layers
    with both reLu and sigmoid activation functions
    
    Here for model 1 to 5 we used the glove200 pretrained embedding (200 stands for the dimension of the word vectors)
    weights=[W] is the argument given to the embedding W is then the matrix built using glove
    
    Also, for all models we used binary_crossentropy as a measure of the loss and
    after testing some other optimizers like adadelta we chose to fit all our models with Adam optimizer
    with default learning rate of 0.001
    """
    
    [train_tweets, labels, test_tweets, nb_tokens, emb_matrix] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}embedding.pkl".format(EMB_DIMS)), "rb"))

    np.random.seed(1)

    model = Sequential()
    model.add(Embedding(nb_tokens, EMB_DIMS, input_length=train_tweets.shape[1], weights=[emb_matrix]))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Fit w/ 0.1 tr/te split
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)

    # Saves the model and predictions
    save_model_predict(model, train, test, "model1")


def model_2(EMB_DIMS, filepath):
    """
    Same as 1 with different seed
    """
    
    [train_tweets, labels, test_tweets, nb_tokens, emb_matrix] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}embedding.pkl".format(EMB_DIMS)), "rb"))

    np.random.seed(2)
    
    model = Sequential()
    model.add(Embedding(nb_tokens, EMB_DIMS, input_length=train_tweets.shape[1], weights=[emb_matrix]))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    # Fit w/ 0.1 tr/te split
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)

    # Saves the model and predictions
    save_model_predict(model, train, test, "model2")
    

def model_3(EMB_DIMS, filepath):
    """
    Same as 1 with different seed
    """
    
    [train_tweets, labels, test_tweets, nb_tokens, emb_matrix] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}embedding.pkl".format(EMB_DIMS)), "rb"))

    np.random.seed(3)
    
    model = Sequential()
    model.add(Embedding(nb_tokens, EMB_DIMS, input_length=train_tweets.shape[1], weights=[emb_matrix]))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())    
    
    
    # Fit w/ 0.1 tr/te split
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)

    # Saves the model and predictions
    save_model_predict(model, train, test, "model3")
    
def model_4(EMB_DIMS, filepath):
    """
    Using keras, we define the first model with one embedding layer and
    one convolutional layer followed by one maxpooling layer and LSTM layer and 2 Dense layers
    with both reLu and sigmoid activation functions
    
    (CONV + LSTM + pretrained Glove200)
    """
    
    [train_tweets, labels, test_tweets, nb_tokens, emb_matrix] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}embedding.pkl".format(EMB_DIMS)), "rb"))

    np.random.seed(4)
    
    model = Sequential()
    model.add(Embedding(nb_tokens, EMB_DIMS, input_length=train_tweets.shape[1], weights=[emb_matrix]))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)

    # Saves the model and predictions
    save_model_predict(model, train, test, "model4")
    
    
def model_5(EMB_DIMS, filepath):
    """
    Model 5 : Embedding + pretraining + LSTM + Dense layer with a sigmoid activation
    """
    [train_tweets, labels, test_tweets, nb_tokens, emb_matrix] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}embedding.pkl".format(EMB_DIMS)), "rb"))

    np.random.seed(5)
    
    model = Sequential()
    model.add(Embedding(nb_tokens, EMB_DIMS, input_length=train_tweets.shape[1], weights=[emb_matrix]))
    model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)

    # Saves the model and predictions
    save_model_predict(model, train, test, "model5")
    
def model_6(NGRAM_RANGE, filepath):
    """
    Model 6 : Embedding + GlobalAveragePooling + Dense layer with a sigmoid activation

    For model 6 to 10, we don't use glove pretrained features but we add the 2-grams
    so we have different set of features
    """
    
    [train_tweets, labels, test_tweets, max_features] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}_gram.pkl".format(NGRAM_RANGE)), "rb"))
    
    np.random.seed(6)
    
    model = Sequential()
    model.add(Embedding(max_features+1, 50, input_length=train_tweets.shape[1]))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)
    
    save_model_predict(model, train, test, "model6")

            
def model_7(NGRAM_RANGE, filepath):
    """
    Model 7 : Embedding + Convolution + Dense layers with sigmoid and reLu activation
    """
    
    [train_tweets, labels, test_tweets, max_features] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}_gram.pkl".format(NGRAM_RANGE)), "rb"))
    
    np.random.seed(7)
    
    model = Sequential()
    model.add(Embedding(max_features+1, 50, input_length=train_tweets.shape[1]))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)
    
    save_model_predict(model, train, test, "model7")


def model_8(NGRAM_RANGE, filepath):
    """
    Model 8 : Embedding + Convolution + MaxPooling+Flattening+ Dense layers with sigmoid and reLu activation
    """
    
    [train_tweets, labels, test_tweets, nb_tokens, max_features] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}_gram.pkl".format(NGRAM_RANGE)), "rb"))
    
    np.random.seed(8)
    
    model = Sequential()
    model.add(Embedding(max_features+1, 20, input_length=train_tweets.shape[1]))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)
    
    save_model_predict(model, train, test, "model8")
    
    
def model_9(NGRAM_RANGE, filepath):
    """
    Model 9: Embedding + Convolution + MaxPooling+LSTM+ Dense layer with sigmoid activation
    """
    
    [train_tweets, labels, test_tweets, max_features] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}_gram.pkl".format(NGRAM_RANGE)), "rb"))
    
    np.random.seed(9)
    
    model = Sequential()
    model.add(Embedding(max_features+1, 50, input_length=train_tweets.shape[1]))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)
    
    save_model_predict(model, train, test, "model9")
    
   
def model_10(NGRAM_RANGE, filepath): 
    """
    Model 10: Embedding + LSTM + Dense layer
    """
    
    [train_tweets, labels, test_tweets, max_features] = \
            cPickle.load(open(os.path.join(filepath, "train_test_{}_gram.pkl".format(NGRAM_RANGE)), "rb"))
    
    np.random.seed(10)
    
    model = Sequential()
    model.add(Embedding(max_features+1, 50, input_length=train_tweets.shape[1]))
    model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    
    model.fit(train_tweets, labels, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)
    train = model.predict_proba(train_tweets, batch_size=128)
    test = model.predict_proba(test_tweets)
    
    save_model_predict(model, train, test, "model10")
    
    
def boost_prediction(model_str):
    """
    Uses xgboost to train a classifier on the train predictions with the expected labels.
    Then, we use this to boost the predicted labels for the test set which improves accuracy.
    
    IN: filename
    OUT: writes the prediction to a file
    """
    
    train_files = [file for file in os.listdir("results/train/") if '2ep_full' in file]
    test_files = [file for file in os.listdir("results/test/") if '2ep_full' in file]
    
    print("Training on {}".format(train_files))
    train = []
    for file in train_files:
        train.append(cPickle.load(open("results/train/{}".format(file), "rb")))
    
    print("Predicting on {}".format(test_files))
    test = []
    for file in test_files:
        test.append(cPickle.load(open("results/test/{}".format(file), "rb")))
    
    train = np.hstack(train)
    test = np.hstack(test)
    
    y = np.array(train.shape[0]//2 * [0] + train.shape[0]//2 * [1])
        
    model = xgb.XGBClassifier().fit(train, y)

    y_pred = model.predict(test)
    y_pred = 2*y_pred - 1
    sub.create_submission(np.arange(1,test.shape[0]+1),y_pred, model_str)    
    
    
def main():
    DATA_PATH = "data/"
    EMB_DIMS = 200
    NGRAM_RANGE = 2
    
    preprocess.main()
    
    train_test_tokenizer.create_embedding()
    
    model_1(EMB_DIMS, DATA_PATH)
    model_2(EMB_DIMS, DATA_PATH)
    model_3(EMB_DIMS, DATA_PATH)
    model_4(EMB_DIMS, DATA_PATH)
    model_5(EMB_DIMS, DATA_PATH)
    
    train_test_tokenizer.create_ngram()
    
    model_6(NGRAM_RANGE, DATA_PATH)
    model_7(NGRAM_RANGE, DATA_PATH)
    model_8(NGRAM_RANGE, DATA_PATH)
    model_9(NGRAM_RANGE, DATA_PATH)
    model_10(NGRAM_RANGE, DATA_PATH)
    
    boost_prediction('fullNN1-5')    


if __name__ == '__main__':
    main()













































