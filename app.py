
import streamlit as st
from transformers import pipeline

st.title('Sentiment Analyser App')
st.write('Welcome to my sentiment analysis app!')


form = st.form(key='sentiment-form')
user_input = form.text_area('Enter your text')
submit = form.form_submit_button('Submit')


classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")


classifier = pipeline("sentiment-analysis")    
result = classifier(user_input)[0]    
label = result['label']    
score = result['score']


if submit:
    classifier = pipeline("sentiment-analysis")
    result = classifier(user_input)[0]
    label = result['label']
    score = result['score']
    if label == 'POSITIVE':
        st.success(f'{label} sentiment')
    else:
        st.error(f'{label} sentiment ')
