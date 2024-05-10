import streamlit as st 
import rag



# Customize the layout
st.set_page_config(page_title="RAG AI", page_icon=":robot:", layout="wide", )     
st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://images.unsplash.com/photo-1509537257950-20f875b03669?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1469&q=80"); 
                    background-attachment: fixed;
                    background-size: cover}}
        </style>
        """, unsafe_allow_html=True)

    
question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?")    




if question:
    response = rag.qa({"question": question})
    st.write(response) 