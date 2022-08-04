import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# from tensorflow.keras.models import load_model
# import plotly.express as px


################# MODEL ##################

# parameters
MODEL='Model_3band_RGB_ha/'
BUCKET='wagon-data-batch913-drought_detection'
STORAGE_LOCATION = f'gs://{BUCKET}/SavedModel/{MODEL}' # GCP path

# load model (cache so it only loads once and saves time)
@st.cache
def load_model():
    return tf.saved_model.load(STORAGE_LOCATION)
    # return hub.load(MODEL)

model = load_model()


################# FUNCTIONS ##################

# preprocessing (possibly move to "utilities.py" file)
def preprocess(text):
    text = text.strip() ## remove whitespaces
    text = text.lower() ## lowercase
    text = ''.join(char for char in text if not char.isdigit()) # remove numbers
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '') # remove punctuation
    return text





################# WEBSITE #################

# # Page appearance
# st.set_page_config(
#     page_title="Pixel Perfect",
#     page_icon="💫",
#     layout="wide",
#     initial_sidebar_state="auto",
# )
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



st.markdown("""
    # Movie Review Sentiment Analyser
    Applying deep learning with Natural Language Processing (NLP) for movie review sentiment analysis
""")



st.header("Let's write a review")

user_text = st.text_area('Add a review:', '''
    ''')

if user_text is not None:

    # Clean reviews
    clean_text = preprocess(user_text)

    # Vectorizing the sentences
    vectorizer = TfidfVectorizer()
    neg_pos_review_vectorized = vectorizer.fit_transform(data.clean_reviews)

    # vectorize new example
    vectorized_example = vectorizer.transform(clean_text) # need vectorizer that was fit on train data
    vectorized_example = pd.DataFrame(vectorized_example.toarray(),
                                        columns = vectorizer.get_feature_names_out())

    # transform vectorized_example & get probability of belonging to topics
    mixture = model.transform(vectorized_example)

    # make into pretty dataframe and print
    topics = pd.DataFrame(mixture)
    topics['review'] = clean_text

    # use predict function
    result = max(topics[0])

    st.title("Here your review after we preprocessed it")
    st.write('Your preprocessed review:', clean_text)


    st.header('Prediction:')
    if result == topics[0][0]:
        st.error("Didn't like the movie")
    elif result == topics[0][1]:
        st.success("Liked the movie")
