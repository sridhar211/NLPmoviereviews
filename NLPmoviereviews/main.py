from NLPmoviereviews.utilities import preprocessing, padding
import gensim.downloader as api
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_score(model,review):
    '''
    Predict review score of user input review
    '''
    word2vec=api.load('glove-wiki-gigaword-100')
    clean_review=preprocessing(review).split()
    clean_review=np.expand_dims(np.asarray(clean_review),axis=0)
    padded=padding(word2vec, clean_review, maxlen=150)
    res=model.predict(padded)
    return res[0,0]

def predict_score_1(model,review):
    '''
    Predict review score of user input review
    '''
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    clean_review=[preprocessing(review).split()]
    review_token = tokenizer.texts_to_sequences(clean_review)
    padded=pad_sequences(review_token, dtype=float, padding='post', maxlen=200)
    res=model.predict(padded)
    return res[0,0]
