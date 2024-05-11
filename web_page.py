import streamlit as st 
import rag

# Customize the layout
st.set_page_config(page_title="RAG AI", page_icon=":robot:", layout="wide", )     

question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?")    

if question:
    response = rag.chain.invoke(question)
    st.write(response) 