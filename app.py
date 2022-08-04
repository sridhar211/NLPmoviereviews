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
MODEL='Model_3band_RGB/'
BUCKET='wagon-data-batch913-drought_detection'
STORAGE_LOCATION = f'gs://{BUCKET}/SavedModel/{MODEL}' # GCP path

# load model (cache so it only loads once and saves time)
@st.cache
def load_model():
    return tf.saved_model.load(STORAGE_LOCATION)
    # return hub.load(MODEL) # alternative way to load model, depending on how/where we save it

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
#     page_title="Movie Review Analyser",
#     page_icon="🍿",
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

user_text = st.text_area('Add your review:', '''
    ''')

if user_text is not None:

    # Clean review
    clean_text = preprocess(user_text)

    # st.write('preprocessed review:', clean_text)

    # vectorize review (need original vectorizer)
    vectorized_review = vectorizer.transform(clean_text) # need vectorizer that was fit on train data
    vectorized_review = pd.DataFrame(vectorized_review.toarray(),
                                        columns = vectorizer.get_feature_names_out())

    # transform vectorized_review & get probability of belonging to positive/negative sentiment
    mixture = model.transform(vectorized_review)

    # make into pretty dataframe and print
    topics = pd.DataFrame(mixture)
    topics['review'] = clean_text

    # most likely sentiment
    result = max(topics[0])

    st.header('Prediction:')
    if result == topics[0][0]:
        st.error("Didn't like the movie")
    elif result == topics[0][1]:
        st.success("Liked the movie")
