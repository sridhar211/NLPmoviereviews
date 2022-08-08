from NLPmoviereviews.utilities import preprocessing, padding
import gensim.downloader as api
import numpy as np

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
