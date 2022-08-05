from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocessing(sentence):
    '''
    Use NLTK to clean text: remove numbers, stop words, and lemmatize verbs and nouns
    '''
    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercasing
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## removing numbers
    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## removing punctuation
    tokenized_sentence = word_tokenize(sentence) ## tokenizing
    stop_words = set(stopwords.words('english')) ## defining stopwords
    tokenized_sentence_cleaned = [w for w in tokenized_sentence
                                  if not w in stop_words] ## remove stopwords
    # 1 - Lemmatizing the verbs
    verb_lemmatized = [WordNetLemmatizer().lemmatize(word, pos = "v")  # v --> verbs
              for word in tokenized_sentence_cleaned]
    # 2 - Lemmatizing the nouns
    noun_lemmatized = [WordNetLemmatizer().lemmatize(word, pos = "n")  # n --> nouns
                for word in verb_lemmatized]
    cleaned_sentence= ' '.join(w for w in noun_lemmatized)
    return cleaned_sentence


def embed_sentence_with_TF(word2vec, sentence):
    '''
    Function to convert a sentence (list of words) into a matrix representing the words in the embedding space
    '''
    embedded_sentence = []
    for word in sentence:
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return np.array(embedded_sentence)


def embedding(word2vec, sentences):
    '''
    Function that converts a list of sentences into a list of matrices
    '''
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence_with_TF(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed

def padding(word2vec, X, maxlen):
    X_embed = embedding(word2vec, X)
    X_pad = pad_sequences(X_embed, dtype='float32', padding='post', maxlen=maxlen)
    return X_pad
