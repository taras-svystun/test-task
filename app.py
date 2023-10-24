__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import torch
import textwrap
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


st.title('Demo for <client> 😇️️️️️️')

uploaded_file = st.file_uploader("Add a text file")
if uploaded_file is not None:
    hf_token = os.environ['HuggingFaceHub_API_Token']

    with open('_sample.txt', 'w') as file:
        file.write("".join([line.decode() for line in uploaded_file]))

    loader = TextLoader('_sample.txt')
    documents = loader.load()

    text_splitter=CharacterTextSplitter(separator='\n',
                                        chunk_size=1000,
                                        chunk_overlap=50)
    text_chunks=text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vectorstore=FAISS.from_documents(text_chunks, embeddings)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                            token=hf_token,
                                            #   low_cpu_mem_usage=True
                                            )

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                token=hf_token,
                                                #  load_in_8bit=True,
                                                #  low_cpu_mem_usage=True
                                                )

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer= tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens = 1024,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                    )

    llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})
    chain =  RetrievalQA.from_chain_type(llm=llm,
                                         chain_type="stuff",
                                         return_source_documents=True,
                                         retriever=vectorstore.as_retriever()
                                         )
    # Try Conversational Retrieval


    with st.form('form'):
        query = st.text_area('Enter your query:', 'What is the Wizard name?')
        submitted = st.form_submit_button('Submit')
        if submitted:
            result=chain({"query": query}, return_only_outputs=True)
            wrapped_text = textwrap.fill(result['result'], width=500)
            st.write(wrapped_text)
