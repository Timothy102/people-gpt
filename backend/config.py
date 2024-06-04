from transformers import AutoTokenizer
from pinecone import Pinecone, ServerlessSpec
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

def create_attention_mask(input_ids):
    # Extract padding token ID (assuming it's set correctly)
    pad_token_id = tokenizer.pad_token_id
    return (input_ids != pad_token_id).float()



# Load dataset (assuming it's a pandas DataFrame)
data = pd.read_csv('data.csv')

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract relevant fields and prepare text for embedding
def prepare_profile_text(profile):
    company_industry = profile.get('company_industry', '')
    experiences = profile.get('experience', '[]')  # Default to empty list if key not present
    inferred_years_experience = profile.get('inferred_years_experience', 0)
    
    # Handle NaN values
    if isinstance(experiences, float) and np.isnan(experiences):
        experiences = '[]'
    
    # Parse experiences JSON string if necessary
    if isinstance(experiences, str):
        try:
            experiences = json.loads(experiences)
        except json.JSONDecodeError:
            experiences = []  # Set to empty list if JSON is invalid
    
    # Combine experience descriptions into a single text
    experience_texts = []
    for exp in experiences:
        if isinstance(exp, dict):
            company_name = exp.get('company', {}).get('name', '')
            role = exp.get('title', '')
            description = exp.get('description', '')
            experience_texts.append(f"{role} at {company_name}: {description}")
    
    combined_experience_text = " ".join(experience_texts)
    profile_text = f"Industry: {company_industry}. Experience: {combined_experience_text}. Inferred Years of Experience: {inferred_years_experience}."
    
    return profile_text

# Prepare profiles for embedding
def create_embeddings(data):
    profiles_texts = []
    profile_ids = data['id'].astype(str).tolist()  # Convert profile_ids to strings

    for _, row in data.iterrows():
        profile = row.to_dict()
        profile_text = prepare_profile_text(profile)
        profiles_texts.append(profile_text)

    # Generate embeddings
    embeddings = model.encode(profiles_texts)
    return embeddings, profile_ids

# Function to create a Pinecone index
def create_pinecone_index(vectors, profile_ids, embeddings):
    pc = Pinecone(api_key="c5b0c09e-b66b-45ef-b7cb-d973dfc6f624", environment="eu-central-1")

    # Create an index in Pinecone (if it doesn't already exist)
    index_name = "quickstart"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,  # Replace with your model dimensions
            metric="euclidean",  # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="eu-central-1"
            )
        )

    # Connect to the index
    index = pc.Index(index_name)
    index.upsert(vectors)
    index_profiles(profile_ids, embeddings)
    return index

# Function to index profiles
def index_profiles(ids, embeddings):
    vectors = []
    for idx, embedding in zip(ids, embeddings):
        vectors.append({
            'id': idx,
            'values': embedding.tolist(),
            'metadata': {'profile_id': idx}
        })

    return vectors

# Function to search profiles based on a natural language query
def search_profiles(query, index, vectors, top_k=10):
    # Encode the query using the same model used for profile embeddings
    query_embedding = model.encode([query])[0]
    
    # Convert query embedding to a list of floats
    query_embedding = query_embedding.tolist()
    
    # Perform semantic search in Pinecone
    try:
        query_result = index.query(vector=[query_embedding], top_k=top_k)
    except Exception as e:
        print("Error from Pinecone:", e)
        return []
    
    # Parse and return results
    results = []
    for match in query_result['matches']:
        profile_id = match['id']
        score = match['score']
        results.append({
            'profile_id': profile_id,
            'score': score
        })
    
    return results