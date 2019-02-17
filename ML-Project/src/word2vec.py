# -*- coding: utf-8 -*-

import numpy as np
import os 
import pickle
from gensim.models import word2vec  
import logging
import implementations as imp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


    
def main():
    DATA_PATH = "../data"

    #Embedding dimension
    N_DIM = 50

    # 200'000 for small data set, 2'500'000 for the full
    N_TWITT = 200000
    
    with open(os.path.join(DATA_PATH, "vocab.pkl"), 'rb') as f:
            vocab = pickle.load(f)
            vocab_size = len(vocab)
    
    train_twitts = imp.import_text('cl_train_pos.txt')
    train_twitts.extend(imp.import_text('cl_train_neg.txt'))
    
    # Creates trains and saves the model
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 
    model = word2vec.Word2Vec(train_twitts, size=N_DIM)
    model.save("../data/word2vec{}.model".format(N_DIM))

    # Testing
    twitt_data = np.zeros((N_TWITT, N_DIM))
    train_files = ['cl_train_pos.txt', 'cl_train_neg.txt']
    for i, file in enumerate(train_files):
        
        with open(os.path.join(DATA_PATH, "twitter-datasets", file), 'rt') as f:
            for l, line in enumerate(f):
                twitt = np.zeros(N_DIM)
                for word in line.strip().split():
                    try:
                        twitt += model.wv[word]
                    except:
                        continue
                twitt_data[i * 100000 + l , :] = twitt
    
    # Create the labels, we know that the first half is positive and the second is negative
    labels = np.concatenate((np.ones(N_TWITT//2), -np.ones(N_TWITT//2)), axis=None)
    
    logistic = LogisticRegression(solver='liblinear')
    
    #get accuracy with a 5 fold cross validation
    print("Accuracy = {:.6}".format(np.mean(cross_val_score(logistic, twitt_data, labels, cv=5, scoring='accuracy'))))
            
if __name__ == '__main__':
    main()
            
            
            
            
            
            
            
            
            
            
            
            