from collections import defaultdict
from typing import Optional
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

import sys
import os
from itertools import compress
import logging

logger = logging.getLogger(__name__)


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


PROMPT = PROMPT
TOKENIZER = None
HIDDEN_STATES = None
EMBEDDINGS_TABLE = None
UMAP_MODEL = None
UMAP_2D = None
TOKEN_IDS = None
ADDITIONAL_POINTS_CACHE = None

@app.on_event("startup")
async def startup_event():
    print("Loading data")
    global PROMPT, TOKENIZER, HIDDEN_STATES, EMBEDDINGS_TABLE, UMAP_MODEL, UMAP_2D, TOKEN_IDS, ADDITIONAL_POINTS_CACHE
    PROMPT, TOKENIZER, HIDDEN_STATES, EMBEDDINGS_TABLE, UMAP_MODEL, UMAP_2D, TOKEN_IDS = load_data()
    ADDITIONAL_POINTS_CACHE = {}
    print("Data loaded")


class Prompt(BaseModel):
    text: Optional[str] = Field(default=PROMPT)
    
class MostSimilarGlobalRequest(BaseModel):
    token_idx: int
    layer_idx: Optional[int] = Field(default=0)
    prompt: Optional[Prompt]
    num_tokens: Optional[int] = Field(default=100)

class MostSimilarGlobalResponse(BaseModel):
    tokens: list[str]
    similarities: list[float]

class GetTokenResponse(BaseModel):
    tokens: list[str]

class GetTokenIdsResponse(BaseModel):
    token_ids: list[int]

class TokenSimilaritiesRequest(BaseModel):
    token_idx: int
    prompt: Prompt
    layer_idx: Optional[int] = Field(default=0)

class TokenSimilaritiesResponse(BaseModel):
    tokens: list[str]
    similarities: list[float]

class CloudRequest(BaseModel):
    sample_rate: float = Field(lt=1, gt = 0, default = 0.1)

class Get2DPointsRequest(BaseModel):
    prompt: Prompt
    layer_idx: Optional[int] = Field(default=0)

class CloudResponse(BaseModel):
    tokens: list[str]
    x: list[float]
    y: list[float]
    total_count: Optional[int] = Field(default=None)

class RootResponse(BaseModel):
    message: str


def tokenize_prompt(token_ids: list[int]) -> dict[str, list]:
    tokens = TOKENIZER.convert_ids_to_tokens(token_ids)
    tokens = [token.replace("▁", " ") for token in tokens]
    return {"tokens": tokens}


def get_2d_cloud_points(sample_rate: float):
    tokens = TOKENIZER.convert_ids_to_tokens(np.arange(len(UMAP_2D)))
    mask = [token is not None for token in tokens]

    # Set sample rate of mask to True
    sampling_mask = np.random.choice([True, False], size=len(tokens), p=[sample_rate, 1-sample_rate])
    # Combine the masks
    mask = np.logical_and(mask, sampling_mask)

    filtered_tokens = list(compress(tokens, mask))
    filtered_x = UMAP_2D[:, 0][mask].tolist()
    filtered_y = UMAP_2D[:, 1][mask].tolist()

    total_count = len(filtered_tokens)
    return dict(
        tokens=filtered_tokens,
        x=filtered_x,
        y=filtered_y,
        total_count=total_count
    )


@app.post("/get_tokens_from_ids", response_model=GetTokenResponse)
def get_tokens_from_ids(token_ids: list[int]) -> GetTokenResponse:
    return TOKENIZER.convert_ids_to_tokens(token_ids)


@app.post("/get_ids_from_tokens", response_model=GetTokenIdsResponse)
def get_ids_from_tokens(tokens: list[str]) -> GetTokenIdsResponse:
    return TOKENIZER.convert_tokens_to_ids(tokens)


@app.post("/token_similarities", response_model=TokenSimilaritiesResponse)
def get_token_similarities(request: TokenSimilaritiesRequest) -> TokenSimilaritiesResponse:
    token_idx = request.token_idx
    layer_idx = request.layer_idx

    if request.prompt.text:
        prompt = request.prompt.text
    else:
        prompt = PROMPT

    token_ids = get_prompt_token_ids(prompt, TOKENIZER)
    if layer_idx == 0:
        prompt_embeddings = get_token_embeddings(token_ids, EMBEDDINGS_TABLE)
    else:
        prompt_embeddings = HIDDEN_STATES.hidden_states[layer_idx].squeeze(0)

    current_emb = prompt_embeddings[token_idx].reshape(1, -1)
    print(f"Comparing token {token_idx} in layer {layer_idx} with {len(prompt_embeddings)} tokens")
    similarities = cosine_similarity_except_self(current_emb, prompt_embeddings).flatten().tolist()
    tokens = tokenize_prompt(TOKENIZER.encode(prompt, add_special_tokens=False))["tokens"]

    return TokenSimilaritiesResponse(
        tokens=tokens,
        similarities=similarities
    )


@app.post("/get_most_similar_global", response_model=MostSimilarGlobalResponse)
def get_most_similar_global(request: MostSimilarGlobalRequest) -> MostSimilarGlobalResponse:
    token_idx = request.token_idx
    layer_idx = request.layer_idx
    num_tokens = request.num_tokens

    if request.prompt.text:
        prompt = request.prompt.text
    else:
        prompt = PROMPT

    token_ids = get_prompt_token_ids(prompt, TOKENIZER)
    if layer_idx == 0:
        prompt_embeddings = get_token_embeddings(token_ids, EMBEDDINGS_TABLE)
    else:
        prompt_embeddings = HIDDEN_STATES.hidden_states[layer_idx].squeeze(0)
    current_emb = prompt_embeddings[token_idx].reshape(1, -1)

    # Calculate similarities with all tokens in the vocabulary
    all_similarities = cosine_similarity(current_emb, EMBEDDINGS_TABLE).flatten()

    # Get the top 50 most similar tokens
    top_k = num_tokens
    sort_idx = np.argsort(all_similarities)[::-1]
    top_indices = sort_idx[1:top_k+1] # exclude the current token
    top_similarities = all_similarities[top_indices]
    tokens = TOKENIZER.batch_decode(top_indices)

    return MostSimilarGlobalResponse(
        tokens=tokens,
        similarities=top_similarities
    )


@app.post("/get_2d_cloud", response_model=CloudResponse)
def get_2d_cloud(request: CloudRequest) -> CloudResponse:
    sample_rate = request.sample_rate
    data = get_2d_cloud_points(sample_rate)
    return CloudResponse(**data)


@app.post("/get_additional_points", response_model=CloudResponse)
def get_additional_points(request: Get2DPointsRequest) -> CloudResponse:
    layer_idx = request.layer_idx

    if request.prompt.text:
        prompt = request.prompt.text
    else:
        prompt = PROMPT

    if ADDITIONAL_POINTS_CACHE.get((prompt, layer_idx), None):
        return CloudResponse(**ADDITIONAL_POINTS_CACHE.get((prompt, layer_idx)))

    token_ids = get_prompt_token_ids(prompt, TOKENIZER)

    if layer_idx == 0:
        prompt_embeddings = get_token_embeddings(token_ids, EMBEDDINGS_TABLE)
    else:
        prompt_embeddings = HIDDEN_STATES.hidden_states[layer_idx].squeeze(0)
    
    points = UMAP_MODEL.transform(prompt_embeddings)
    tokens = tokenize_prompt(TOKENIZER.encode(prompt, add_special_tokens=False))["tokens"]

    ADDITIONAL_POINTS_CACHE[(prompt, layer_idx)] = dict(
        tokens=tokens,
        x=points[:, 0],
        y=points[:, 1],
        total_count=len(tokens)
    )

    return CloudResponse(**ADDITIONAL_POINTS_CACHE[(prompt, layer_idx)])


@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(message="Welcome to the Phi-3 Transformer Visualization API")

@app.on_event("shutdown")
async def shutdown_event():
    print("Application is shutting down")

