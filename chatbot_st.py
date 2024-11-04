from langchain.llms import OpenAI  # Correct import for OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY']= os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2']='true'

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that converses with a patient to help diagnose the disease."),
        ("user", "Question: {question}")
    ]
)

st.title("How can I help?")
input_text = st.text_input("Let's chat")

# Initialize OpenAI LLM with the correct API key
llm = OpenAI(openai_api_key=openai_api_key)

# Create an LLMChain without a custom output parser
#chain = LLMChain(llm=llm, prompt=prompt)
#chain = LLMChain(llm=llm, prompt=prompt)
chain = prompt|llm
# Run the chain if input is provided
if input_text:
    result = chain.invoke({'question': input_text})
    st.write(result)
