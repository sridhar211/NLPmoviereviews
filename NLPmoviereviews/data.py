import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import text_to_word_sequence

def load_data(percentage_of_sentences=10):
    '''
    Load the imdb_reviews dataset for given percentage of the dataset.
    Returns train-test sets
    X--> returned as list of words in lower case
    y--> returned as two classes 0 and 1 for bad and good reviews
    '''
    train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], batch_size=-1, as_supervised=True)

    train_sentences, y_train = tfds.as_numpy(train_data)
    test_sentences, y_test = tfds.as_numpy(test_data)

    # Take only a given percentage of the entire data
    if percentage_of_sentences is not None:
        assert(percentage_of_sentences> 0 and percentage_of_sentences<=100)

        len_train = int(percentage_of_sentences/100*len(train_sentences))
        train_sentences, y_train = train_sentences[:len_train], y_train[:len_train]

        len_test = int(percentage_of_sentences/100*len(test_sentences))
        test_sentences, y_test = test_sentences[:len_test], y_test[:len_test]

    X_train = [text_to_word_sequence(_.decode("utf-8")) for _ in train_sentences]
    X_test = [text_to_word_sequence(_.decode("utf-8")) for _ in test_sentences]

    return X_train, y_train, X_test, y_test


def load_data_sent(percentage_of_sentences=10):
    '''
    Load the imdb_reviews dataset for given percentage of the dataset.
    Returns train-test sets
    X--> returned as sentences in lower case
    y--> returned as two classes 0 and 1 for bad and good reviews
    '''
    X_train, y_train, X_test, y_test = load_data(percentage_of_sentences)
    X_train = [' '.join(_) for _ in X_train]
    X_test = [' '.join(_) for _ in X_test]
    return X_train, y_train, X_test, y_test
