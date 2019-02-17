# -*- coding: utf-8 -*-

import numpy as np
import os 
import _pickle as cPickle
from gensim.models import word2vec  
import logging
import implementations as imp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


def main():
    
    DATA_PATH = "../data/"

    #Embedding dimension
    N_DIM = 200

    # 200'000 for small data set, 2'500'000 for the full
    N_TWITT = 200000

    #chose the file to read contains the training sets as well as the testing set for valuation
    file_name ='for_graph_bigram.pkl'


    train_tweets, labels, test_tweets, nb_tokens = \
            cPickle.load(open(os.path.join(file_name), mode='rb'))

    train_data = []    
    for i in range(train_tweets.shape[0]):
        str_tweet = []
        for j in range(train_tweets.shape[1]):
            if train_tweets[i, j] != 0:
               str_tweet.append(str(train_tweets[i, j]))
        
        train_data.append(str_tweet)
           
    # Creates trains and saves the model 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
    model = word2vec.Word2Vec(train_data, size=N_DIM)
    model.save("../data/word2vec_ngram.model")

    # Testing
    emb_tweet = np.zeros((len(train_data), N_DIM))
    for i, tweet in enumerate(train_data):
        out_tweet = np.zeros(N_DIM)
        for word in tweet:
            try:
                out_tweet += model.wv[word]
            except KeyError:
                pass

        emb_tweet[i, :] = out_tweet

    #only keep the training tweets
    train_emb_tweet = emb_tweet[:N_TWITT]

    # Create the labels, we know that the first half is positive and the second is negative
    labels = np.concatenate((np.ones(N_TWITT//2), -np.ones(N_TWITT//2)), axis=None)


    print("Fitting...")
    logistic = LogisticRegression(solver='liblinear')

    #get accuracy with a 5 fold cross validation
    print("Accuracy = {:.6}".format(np.mean(cross_val_score(logistic, train_emb_tweet, labels, cv=5, scoring='accuracy'))))
    
    
if __name__ == '__main__':
    main()





































