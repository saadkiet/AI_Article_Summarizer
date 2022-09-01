import streamlit
import pandas as pd
from transformers import pipeline
import streamlit as st
import numpy as np

from bs4 import BeautifulSoup
import requests




st.title("AI News/Blog Summarizer 🤓")
st.markdown("Enter URL of the News/Blog post in the empty field below")


@st.cache(allow_output_mutation=True)
def get_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = get_model()


#user_input = st.text_area('Enter URL ')
ULR_=st.text_input("Enter ULR")
button = st.button("Summarize")

if ULR_ and button:
    st.success("Summarizing the article, please wait...")
    r = requests.get(ULR_)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all(['h1', 'p'])
    text = [result.text for result in results]
    ARTICLE = ' '.join(text)
    ARTICLE = ARTICLE.replace('.', '.<eos>')
    ARTICLE = ARTICLE.replace('?', '?<eos>')
    ARTICLE = ARTICLE.replace('!', '!<eos>')

    max_chunk = 500
    sentences = ARTICLE.split('<eos>')
    current_chunk = 0
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            print(current_chunk)
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])

    res = summarizer(chunks, max_length=150, min_length=30, do_sample=False)
    text = ' '.join([summ['summary_text'] for summ in res])
    st.write(text)


    #res = generator(user_input, max_length=500, do_sample=True, temperature=0.9)
    #st.write(res[0]['generated_text'])

