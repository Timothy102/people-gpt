from typing import Dict, List
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Candidate API", version="1.0")

# Example data for candidates
candidates_data = {
    "python": ["Alice", "Bob", "Charlie"],
    "java": ["David", "Eve", "Frank"]
}

@app.get("/")
async def root():
    return {"message": "Welcome to the Candidate API!"}

@app.post("/get_candidates")
async def get_candidates(candidate_request: str):
    """
    Returns a list of possible candidates based on the provided criteria.
    
    Parameters:
    - candidate_request: A dictionary containing the criteria for selecting candidates.
    """
    criteria = candidate_request.get("criteria", "").lower()
    if criteria in candidates_data:
        candidates = candidates_data[criteria]
    else:
        candidates = ["No candidates found for the given criteria."]
    return {"candidates": candidates}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080)
