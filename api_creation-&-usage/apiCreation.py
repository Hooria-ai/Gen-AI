from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama
from langchain.schema import HumanMessage

# Get the OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A Simple API server"
)

# Initialize the OpenAI Chat LLM
openai_llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")

# Initialize the Ollama LLM
ollama_llm = Ollama(model="llama2")

# Request body model for the API
class RequestData(BaseModel):
    topic: str

# API endpoint for essay generation using OpenAI
@app.post("/essay")
async def generate_essay(data: RequestData):
    prompt = f"Write an essay about {data.topic} of 100 words!"

    try:
        response = openai_llm([HumanMessage(content=prompt)])
        essay = response.content
        return {"essay": essay}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating essay: {str(e)}")

# API endpoint for poem generation using Ollama's LLaMA2 model
@app.post("/poem")
async def generate_poem(data: RequestData):
    prompt = f"Write a poem about {data.topic} of 100 words!"

    try:
        poem = ollama_llm(prompt)
        return {"poem": poem}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating poem: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)