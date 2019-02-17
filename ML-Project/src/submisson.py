# -*- coding: utf-8 -*-

import os
import numpy as np

def get_test_data():
    """
    Get the tweets from the test data text file and returns them and their corresponding 
    labels as arrays.
    """
    ids = []
    test_tweets = []

    with open("../data/twitter-datasets/cl_test_data.txt", mode='rt') as f:
        for line in f : 
            ids.append(line.strip().split(',')[0])
            test_tweets.append(' '.join(line.strip().split()[1:]))    
    
    return ids, test_tweets


def create_submission(ids, y_pred, descript_str=''):
    """
    From predictions on test set, creates a file for submission on CrowdAI
    """
    with open("results/submission_{}.txt".format(descript_str), mode='wt') as f:
        f.write("Id,Prediction\n")
        for i in range(ids.shape[0]):
            f.write(str(ids[i])+','+str(int(y_pred[i]))+'\n')

















































