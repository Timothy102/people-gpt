# PartnerHQ TimC Take Home Task: "PeopleGPT"

### What I've Done

Hey, Stan. I used Mistral-7B for NaiveRAG(query + retrieval) & Self-Query RAG to try and build a “people search engine”. The majority of the work can be found in both notebooks. You’ve been added as a private collaborator to the private repo here.

## Running the Frontend
To run the frontend components of the "PeopleGPT" project, follow these steps:

1. **Frontend:**
   - Navigate to the `frontend` directory.
   - Run `npm install` to install dependencies.
   - Run `npm start` to start the frontend server.

2. **Backend:**
   - Navigate to the `backend` directory.
   - Run `pip install -r requirements.txt` to install Python dependencies.
   - Run `python app.py` to start the backend server.

## Running the Backend
To run the backend components of the "PeopleGPT" project, follow these steps:

2. **Backend:**
   - Navigate to the `backend` directory.
   - Run `pip install -r requirements.txt` to install Python dependencies.
   - Run `python app.py` to start the backend server.

### Conclusive Thoughts

---

1. GPU!!!
    1. Mistral-7B takes on average 240s to generate a from-query response.
    2. We should look into smaller models and/or fine-tuned versions. The biggest hurdle in the current infra. 
2. RAG
    1. With the data being so well-defined and granular, we can perform much better filtering, i.e heuristic rules & Regex.
    2. We ought to look for hierarchical indices filtering, where we filter based on summaries of vector indices, not the indices themselves.
3. Scale
    1. The two bottlenecks in this infra:
        1. Model retrieving
            1. Here, we can reduce the amount of data it has to process(pure heuristic/regex filtering) + hierarchical indicies(summaries 1st).
        2. Model generating
            1. Here, we have to get the smallest possible model.
            2. Then scale vertically:
                1. More powerful GPUs.
                2. DL methods: pruning, distillation, quantization.
                3. K8s vertical scaler across worker nodes.
                4. Model partitioning.
                5. Inference servers, like Triton.
4. Comments
    1. Current implementation is way too memory & compute intensive for what we need. 
    2. It’s been a packed week in SF and back. This took me a day. It’s far from perfect. I would have wanted to spend more time on this, coming up with a cleaner solution to perform filtering + RAG + fine-tuning. Unfortunately, that’s all the time I have at the moment.
    3. Feel free to take this code and use it in whatever way.

tim@timcvetko.com