__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator


openai_api_key = st.sidebar.text_input('OpenAI API Key')

st.title('Demo for <client> 😇️️️️️️')

uploaded_file = st.file_uploader("Add a text file")
if uploaded_file is not None and openai_api_key != '':
    os.environ["OPENAI_API_KEY"] = openai_api_key

    with open('_sample.txt', 'w') as file:
        file.write("".join([line.decode() for line in uploaded_file]))

    loader = TextLoader("_sample.txt")
    documents = loader.load()

    index = VectorstoreIndexCreator().from_loaders([loader])

    with st.form('form'):
        query = st.text_area('Enter your query:', 'What is the Wizard name?')
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write(index.query(query))
