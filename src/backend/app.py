from typing import Optional
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

import sys
import os

# Add the current directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from phi35.utils import (
    get_prompt_token_ids,
    get_token_embeddings_table,
    load_tokenizer,
    get_hidden_states,  
    get_umap_model, 
    get_2d_representation,
    get_token_embeddings,
    cosine_similarity,
    PROMPT
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_data():
    tokenizer = load_tokenizer()
    hidden_states = get_hidden_states(PROMPT, tokenizer)
    embeddings_table = get_token_embeddings_table()
    umap_model = get_umap_model(embeddings_table)
    umap_2d = get_2d_representation(embeddings_table, umap_model)
    token_ids = get_prompt_token_ids(PROMPT, tokenizer)
    return PROMPT, tokenizer, hidden_states, embeddings_table, umap_model, umap_2d, token_ids


def cosine_similarity_except_self(emb, embeddings):
    prompt_similarities = cosine_similarity(emb, embeddings)
    one_mask = np.isclose(prompt_similarities, 1, atol=1e-3)
    non_one_max = np.max(prompt_similarities[~one_mask])
    prompt_max_sim = non_one_max
    prompt_min_sim = np.min(prompt_similarities)
    new_similarities = np.clip(prompt_similarities, prompt_min_sim, prompt_max_sim)
    normalized_similarities = (new_similarities - prompt_min_sim) / (prompt_max_sim - prompt_min_sim)
    return normalized_similarities


PROMPT, TOKENIZER, HIDDEN_STATES, EMBEDDINGS_TABLE, UMAP_MODEL, UMAP_2D, TOKEN_IDS = load_data()


class Prompt(BaseModel):
    text: Optional[str] = Field(default=PROMPT)
    
class MostSimilarGlobalResponse(BaseModel):
    tokens: list[str]
    similarities: list[float]

class MostSimilarGlobalRequest(BaseModel):
    token_idx: int
    prompt: Prompt

class TokenizeResponse(BaseModel):
    tokens: list[str]

class TokenSimilaritiesResponse(BaseModel):
    tokens: list[str]
    similarities: list[float]

class RootResponse(BaseModel):
    message: str

class TokenSimilaritiesRequest(BaseModel):
    token_idx: int
    prompt: Prompt

def tokenize_prompt(token_ids: list[int]) -> dict[str, list]:
    tokens = TOKENIZER.convert_ids_to_tokens(token_ids)
    # tokens = [token.replace("▁", " ") for token in tokens if token != "<0x0A>"]
    return {"tokens": tokens}


@app.post("/tokenize", response_model=TokenizeResponse)
def tokenize(prompt: Optional[Prompt]) -> TokenizeResponse:
    if prompt.text:
        prompt = prompt.text
    else:
        prompt = PROMPT
    token_ids = TOKENIZER.encode(prompt, add_special_tokens=False)
    return TokenizeResponse(**tokenize_prompt(token_ids))


@app.post("/token_similarities", response_model=TokenSimilaritiesResponse)
def get_token_similarities(request: TokenSimilaritiesRequest) -> TokenSimilaritiesResponse:
    token_idx = request.token_idx
    prompt = request.prompt.text if request.prompt.text else PROMPT

    token_ids = get_prompt_token_ids(prompt, TOKENIZER)
    prompt_embeddings = get_token_embeddings(token_ids, EMBEDDINGS_TABLE)

    current_emb = prompt_embeddings[token_idx].reshape(1, -1)
    
    # Calculate similarities
    similarities = cosine_similarity_except_self(current_emb, prompt_embeddings).flatten().tolist()
    
    tokens = tokenize_prompt(TOKENIZER.encode(prompt, add_special_tokens=False))["tokens"]
    
    return TokenSimilaritiesResponse(
        tokens=tokens,
        similarities=similarities
    )

@app.post("/get_most_similar_global", response_model=MostSimilarGlobalResponse)
def get_most_similar_global(request: TokenSimilaritiesRequest) -> MostSimilarGlobalResponse:
    token_idx = request.token_idx
    prompt = request.prompt.text if request.prompt.text else PROMPT

    token_ids = get_prompt_token_ids(prompt, TOKENIZER)
    prompt_embeddings = get_token_embeddings(token_ids, EMBEDDINGS_TABLE)
    current_emb = prompt_embeddings[token_idx].reshape(1, -1)

    # Calculate similarities with all tokens in the vocabulary
    all_similarities = cosine_similarity(current_emb, EMBEDDINGS_TABLE).flatten()

    # Get the top 50 most similar tokens
    top_k = 50
    top_indices = nlargest(top_k, range(len(all_similarities)), key=all_similarities.__getitem__)
    top_similarities = [all_similarities[i] for i in top_indices]
    top_tokens = [TOKENIZER.decode([i]) for i in top_indices]

    return MostSimilarGlobalResponse(
        tokens=top_tokens,
        similarities=top_similarities
    )


@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(message="Welcome to the Phi-3 Transformer Visualization API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)