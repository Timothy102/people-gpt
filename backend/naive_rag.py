from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import create_attention_mask, search_profiles

import os
import torch

# MPS Enable on Torch
os.environ["TORCH_MPS_ENABLED"] = "0"

# Log in to Hugging Face
login(token="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")

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = FastAPI(title="NaiveRAG API", version="1.0")

# CORS middleware for handling Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the PartnerHQ NaiveRAG Mistral-7B API!"}

@app.post("/query")
async def perform_pipeline(query: str):
    """
    Performs the pipeline for the given query.
    """
    context = "Query: " + query + "\nSearch Results:\n"
    
    # Search for profiles across the embedding space
    results = search_profiles(query)
    for result in results:
        context += f"Profile ID: {result['profile_id']}, Score: {result['score']}\n"
    
    # Tokenize the context with padding and truncation
    inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=200)

    # Create the attention mask
    attention_mask = create_attention_mask(inputs['input_ids'])

    # Generate text with attention mask
    output = model.generate(input_ids=inputs['input_ids'], attention_mask=attention_mask, max_length=200, max_new_tokens=200)
    return {"output": output}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)