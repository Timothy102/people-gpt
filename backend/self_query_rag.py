from pinecone import Pinecone, PodSpec

# Langchain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
    AttributeInfo
)
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.pinecone import PineconeTranslator
from langchain_openai import (
    ChatOpenAI, 
    OpenAIEmbeddings
)
from langchain_pinecone import PineconeVectorStore
from langchain.indexes import SQLRecordManager, index
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

@app.post("/recommend-candidates")
async def recommend_candidates(job_description: str, context: str):
    def format_docs(docs):
        return "\n\n".join(f"{doc.page_content}\n\nMetadata: {doc.metadata}" for doc in docs)

    chat_model = ChatOpenAI(
        model='gpt-3.5-turbo-0125',
        temperature=0,
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
                Your goal is to recommend candidates based on their qualifications and the provided job description. 
                If a candidate doesn't seem relevant, omit them from your response. Never refer to candidates that are not in your context. 
                If you cannot recommend any candidates, suggest refining the job description or search criteria. 
                You cannot recommend more than five candidates. Your recommendation should be relevant, specific, and at least two to three sentences long.

                YOU CANNOT RECOMMEND A CANDIDATE IF THEY DO NOT APPEAR IN YOUR CONTEXT.

                # TEMPLATE FOR OUTPUT
                - [Candidate Name](profile link):
                    - Relevant Experience:
                    - Skills:
                    - (Your reasoning for recommending this candidate)
                
                Job Description: {job_description} 
                Context: {context} 
                """

            ),
        ]
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | chat_model
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    output = {}
    curr_key = None
    for chunk in rag_chain_with_source.stream(question):
        for key in chunk:
            if key not in output:
                output[key] = chunk[key]
            else:
                output[key] += chunk[key]
            if key != curr_key:
                print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
            else:
                print(chunk[key], end="", flush=True)
            curr_key = key

    return output


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080)
