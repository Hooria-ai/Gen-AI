import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post("http://127.0.0.1:8000/essay",
    json={'topic': input_text})
    if response.status_code == 200:
        return response.json()['essay']
    else:
        return f"Error: {response.status_code} - {response.text}"

def get_ollama_response(input_text):
    response = requests.post("http://127.0.0.1:8000/poem",
    json={'topic': input_text})
    if response.status_code == 200:
        return response.json()['poem']
    else:
        return f"Error: {response.status_code} - {response.text}"

st.title("Langchain Demo with LLaMA2 and GPT-4")

st.header("Essay Generator (GPT-4)")
input_text1 = st.text_input("Enter a topic for an essay:")

if input_text1:
    st.write("Generating essay...")
    essay = get_openai_response(input_text1)
    st.write(essay)

st.header("Poem Generator (LLaMA2)")
input_text2 = st.text_input("Enter a topic for a poem:")

if input_text2:
    st.write("Generating poem...")
    poem = get_ollama_response(input_text2)
    st.write(poem)