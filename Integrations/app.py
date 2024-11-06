from langchain_openai import OpenAI  # Correct import for OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
#import faiss
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY']= os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2']='true'

pdf = PyPDFLoader("/home/hooria.najeeb@vaival.tech/miniconda3/envs/LC/TASKS/Atopic-Dermatitis.pdf")
page = pdf.load_and_split()
vectorstore = FAISS.from_documents(page, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()




# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a clinical assistant with 5+ years of experience. "
                    "you will  conversate with a patient to provide the patient with information about the disease, its symptoms, and how to manage it."
                    "particularly for atopic dermatitis"
                    "you can ask questions about the patient feellings and symptoms" 
                    "\n\n"
                    "{context}"),
        ("user", "Question: {question}"),
        #MessagesPlaceholder("chat_history"),
    ]
)







# Sample context (this can be replaced with dynamic context from previous interactions)
context = "Patient mentioned having dry and itchy skin for the past two weeks."

# Run the chain if input is provided



st.title("Atopoic Dermatitis Chatbot")
input_text = st.text_input("Let's solve your problem")



# Initialize OpenAI LLM with the correct API key
llm = OpenAI(openai_api_key=openai_api_key)

chain = prompt|llm
# Run the chain if input is provided
if input_text:
    result = chain.invoke({'question': input_text, 'context': context})
    st.write(result)

