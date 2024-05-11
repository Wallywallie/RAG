import streamlit as st 
import rag
import time

# Customize the layout
st.set_page_config(page_title="RAG AI", page_icon=":robot:", layout="wide", )     

question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?")    

if question:
    
    t1 = time .perf_counter()
    response = rag.chain.invoke(question)
    t2 = time .perf_counter()
    print(f"Generate answer: {t2-t1}")

    st.write(response) 
    t3 = time .perf_counter()
    print(f"writing answer: {t3-t2}")