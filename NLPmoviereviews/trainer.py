from NLPmoviereviews.model import create_model, fit_model, save_model
from NLPmoviereviews.data import load_data_sent
from NLPmoviereviews.utilities import preprocessing
import gensim.downloader as api
from NLPmoviereviews.utilities import padding
import pickle


if __name__=='__main__':
    '''
    Train the model on dataset and save the trained model
    '''
    # Load data set
    X_train, y_train, X_test, y_test = load_data_sent(percentage_of_sentences=100)
    # Preprocess the data set
    X_train = [preprocessing(_) for _ in X_train]
    X_train = [_.split() for _ in X_train]
    X_test = [preprocessing(_) for _ in X_test]
    X_test = [_.split() for _ in X_test]
    # Load pretrained vectorisation
    word2vec=api.load('glove-wiki-gigaword-100')
    # Pad the train test set
    pad_length=150
    X_train_pad = padding(word2vec, X_train, maxlen=pad_length)
    X_test_pad = padding(word2vec, X_test, maxlen=pad_length)
    # Create and train the model
    model=create_model()
    nlp_model=fit_model(model, X_train_pad, y_train)
    # Check accuracy on test set
    res = nlp_model.evaluate(X_test_pad, y_test, verbose=0)
    print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')
    save_model(nlp_model)
