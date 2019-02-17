# -*- coding: utf-8 -*-
import os
import re
from nltk.stem.porter import PorterStemmer
import itertools

def load_dicts(DICT_PATH):
    """
    Loads the dictionaries for the most common spelling errors
    """
    
    dict_typo = {}
    with open(os.path.join(DICT_PATH, "emnlp_dict.txt"), mode='rt') as f:
        for line in f:
            key, value = line.rstrip('\n').split('\t')
            dict_typo.update({key:value})

    with open(os.path.join(DICT_PATH, "Test_Set_3802_Pairs.txt") , mode='r') as f:
        for line in f:
            try:
                key, value = line.rstrip('\n').split('\t')[1].split(' | ')
                dict_typo.update({key:value})
            # Some values have multiple keys affected to them
            except ValueError:
                ls = line.rstrip('\n').split('\t')[1].split(' | ')
                key = ls[0]
                value= ls[1]
                    # Update dict with all the keys
                dict_typo.update({key:value})
                    
    with open(os.path.join(DICT_PATH,  "typo-corpus-r1.txt"), mode='rt') as f:
        for line in f:
            key, value, _, _, _, _ = line.rstrip('\n').split('\t')
            dict_typo.update({key:value})
           
    return dict_typo


def remove_repetitions(tweet):
    """
    Functions that remove noisy character repetition like for instance :
    llloooooooovvvvvve ====> love
    This function reduce the number of character repetition to 2 and checks if the word belong the english
    vocabulary by use of pyEnchant and reduce the number of character repetition to 1 otherwise
    Arguments: tweet (the tweet)    
    """
    
    tweet = tweet.split()
    for i in range(len(tweet)):
        tweet[i] = ''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i])).replace('#', '')
    tweet=' '.join(tweet)
    return tweet


def clean_tweets(filename, in_path, out_path, dict_typo, only_words=False, stemmer=None, min_len=None):
    """
    Cleans the original tweet data and rewrites the cleaned tweets to new files. This function is modular and can be used to treat the tweets in many ways
    that we ended up not using for accuracy reasons. 
    
    IN: Paths of the data and the name of the files we want to clean, kwarg to selects the processing type.
    
    OUT: Writes the cleaned data to the specified outpath.
    """
    
    print("Cleaning with: only words={}, stemmer={}, minimal length={}".format(only_words, stemmer!=None, min_len))    
    with open(os.path.join(in_path, filename), mode='rt', encoding='utf-8') as rf:
        with open(os.path.join(out_path, 'cl_'+filename), mode='wt', encoding='utf-8') as wf:
            
            for line in rf:
                if 'test' in filename:
                    ID = line.strip().split(',')[0]+','
                    tweet = ' '.join(line.strip().split()[1:])
                else:
                    ID = ''
                    tweet =  line.strip()
                    
                remove_repetitions(tweet)
                
                # Spell checker for commonly missspeled words
                tweet = tweet.strip().split()
                for i, word in enumerate(tweet):        
                    if word in dict_typo.keys():
                        tweet[i] = dict_typo[word]  
                        
                tweet = ' '.join(tweet)
                    
                tweet = re.sub(r"\'s", " \'s", tweet)
                tweet = re.sub(r"\'ve", " \'ve", tweet) 
                tweet = re.sub(r"n\'t", " n\'t", tweet)
                tweet = re.sub(r" ca ", " can ", tweet)
                tweet = re.sub(r"\'re", " \'re", tweet)
                tweet = re.sub(r"\'d", " \'d", tweet)
                tweet = re.sub(r"\'l", " \'ll", tweet)
                tweet = re.sub(r"\'ll", " \'ll", tweet)
                tweet = re.sub(r"\s{2,}", " ", tweet)
                tweet = re.sub(r'<([^>]+)>', ' ',tweet)         # Removes usr and url
                tweet = re.sub(r'^#| #', ' ', tweet)                            # Removes hashtags
                tweet = re.sub(r'\d+(x)\d+', '<img>', tweet)                    # Removes picture frames            
                
                if only_words:
                    tweet = re.sub(r'[^a-z]', ' ', tweet)                       # Only keeps words
                 
                tweet = tweet.strip().split()
                
                if stemmer != None:
                    tweet = [stemmer.stem(word) for word in tweet]              # stemming             
                
                if min_len is not None:
                    wf.write(ID+' '.join([word for word in tweet if len(word) >= min_len])+'\n')
                else:
                    wf.write(ID+' '.join(tweet)+'\n')
                    

def main():

    DICT_PATH = "dict/"
    OR_TWITT_PATH = "data/twitter-datasets-original"
    NEW_TWITT_PATH = "data/twitter-datasets"
    FULL = True 
    
    dict_typo = load_dicts(DICT_PATH)

    if FULL:
        files = [i for i in os.listdir(OR_TWITT_PATH) if i.endswith('.txt')]        
    else:
        files = [i for i in os.listdir(OR_TWITT_PATH) if not i.endswith('full.txt')]
    
    stemmer = PorterStemmer()
    
    for file in files:
        print("Processing {} ...".format(file))
        clean_tweets(file, OR_TWITT_PATH, NEW_TWITT_PATH, dict_typo, only_words=False, stemmer=None, min_len=None)
                        
        
if __name__ == '__main__':
    main()





















