

from langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser


# Load environment variables
load_dotenv(".env")

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'

# Load PDF with error handling
pdf = PyPDFLoader("Celiac.pdf")
page = pdf.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
page = text_splitter.split_documents(page)

vectorstore = FAISS.from_documents(page, OpenAIEmbeddings())
output = StrOutputParser()
retriever = vectorstore.as_retriever()

# Initialize the LLM and memory
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

# # Initialize chat history
chat_history = []


# Define the prompt template with MessagesPlaceholder for chat history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a clinical assistant with 5+ years of experience. "
                    "You will converse with a patient to provide information about the disease, its symptoms, and management strategies for celiac disease. "
                    "You can ask questions about the patient's feelings and symptoms."
                    "\n\n"
                    "{context}"),
        #MessagesPlaceholder({{}}),
        MessagesPlaceholder(variable_name="chat_history"), 
        ("user", "Question: {question}"),
    ]
)



# Sample context
context = "Patient mentioned that after consuming even a small amount of gluten, he suffered from rashes and vomiting."

# Streamlit UI
st.title("Celiac Chatbot")
input_text = st.text_input("Let's solve your problem")

# Initialize OpenAI LLM with the correct API key
llm = OpenAI(openai_api_key=openai_api_key)


# Combine prompt and LLM
#chain = LLMChain(prompt=prompt, llm=llm)
chain = prompt | llm 
# Run the chain if input is provided
if input_text:
    # Make sure chat_history is passed as a variable to `invoke`
    result = chain.invoke({'chat_history': chat_history,'question': input_text, 'context': context})
    st.write(result)

    # Append the question and response to the chat history
    chat_history.append({"role": "user", "content": input_text})
    chat_history.append({"role": "assistant", "content": result})

