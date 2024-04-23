import pandas as pd
import numpy as np
import streamlit as st 
import joblib
import spacy

st.sidebar.header('Please chosse the menu')
st.sidebar.selectbox(options=['Select the emotions example','sad','happy','angry'],label='selectbox')
st.sidebar.button('sumbit')

new_text=st.text_input('Enter your text')


file=joblib.load("Human_emotion_new_project.pkl")
Button=st.button('Enter')

def pre_procsses(text):
    nlp=spacy.load('en_core_web_sm')
    doc=nlp(text)
    proccessed_token= [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    pre_procsses=' '.join(proccessed_token)
    return pre_procsses                  

if Button==True:
    new_text=pre_procsses(new_text)
    st.write(new_text)
    result=file.predict([new_text])
    st.write(result)