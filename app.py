import streamlit as st
import numpy as np
import pandas as pd
import gensim.downloader as api
import tensorflow as tf

from tensorflow.keras.models import load_model
from NLPmoviereviews.main import predict_score_1


################# MODEL ##################

# parameters
MODEL='saved_model/nlp_1_model/'
# MODEL='saved_model/bert_model/'


# load model (cache so it only loads once and saves time)
#@st.cache
def load_model_cache():
    model=load_model(MODEL)
    return model

model = load_model_cache()


################# WEBSITE #################

# Page appearance
st.set_page_config(
    page_title="Movie Review Analyser",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="auto",
)

# CSS Styling changes
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("""
    # Movie Review Sentiment Analyser
    Applying deep learning with Natural Language Processing (NLP) for movie review sentiment analysis
""")



st.header("Let's write a review")

user_text = st.text_area('Add your review:', ''' The best movie ever
    ''')

if user_text is not None:

    result = predict_score_1(model, user_text)
    # result = tf.sigmoid(model(tf.constant(user_text)))

    # display sentiment
    st.header('Prediction:')
    if result <= 0.5:
        st.error("Didn't like the movie")
    elif result > 0.5:
        st.success("Liked the movie")
