# Using Gemini
# Using text file as knowledge base

import os
import streamlit as st
#import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import traceback

#openai_api_key = os.getenv("OPENAI_API_KEY")
#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
google_api_key = os.getenv("GOOGLE_API_KEY")

# Simple Streamlit app
st.set_page_config(page_title="Strata Property in Malaysia", page_icon=":house:", layout="wide")
st.title("Ask about Strata Property in Malaysia")
query = st.text_input("Enter your question")

# If submit button click
if st.button("Submit"):
    if not query.strip():
        st.error("Please enter a query")
    else:
        try:
            # Define custom LLM Model
            llm_predictor = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7)

            # Load documents from directory name 'data'
            documents = SimpleDirectoryReader('data').load_data()
            service_context = ServiceContext.from_defaults()

            # Build Index
            index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            st.success(response)
        except Exception as e:
            st.error(f"Error occurred: {e}")
            st.error(traceback.format_exc())
