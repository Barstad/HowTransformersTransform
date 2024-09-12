from typing import Literal, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
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

from model.utils import (
    get_2d_representation,
    get_prompt_token_ids,
    get_token_embeddings,
    get_token_embeddings_table,
    get_tokenizer,
    get_hidden_states,  
    get_umap_model, 
    cosine_similarity,
    PROMPT,
    SMALL_MODEL,
    LARGER_MODEL
)


def load_data(model_name:str):
    tokenizer = get_tokenizer(model_name)
    hidden_states = get_hidden_states(model_name)
    input_embeddings_table = get_token_embeddings_table(model_name, "input")
    output_embeddings_table = get_token_embeddings_table(model_name, "output")
    token_ids = get_prompt_token_ids(PROMPT, tokenizer)
    # umap_model = get_umap_model(model_name)
    
    # embed_2d = get_2d_representation(model_name, umap_model, embeddings_table, load=True)

    attributes = {
        "prompt": PROMPT,
        "tokenizer": tokenizer,
        "hidden_states": hidden_states,
        "input_embeddings_table": input_embeddings_table,
        "output_embeddings_table": output_embeddings_table,
        # "umap_model": umap_model,
        "token_ids": token_ids,
        # "embed_2d": embed_2d
    }

    return attributes


def cosine_similarity_except_self(emb, embeddings):
    prompt_similarities = cosine_similarity(emb, embeddings)
    one_mask = np.isclose(prompt_similarities, 1, atol=1e-3)
    non_one_max = np.max(prompt_similarities[~one_mask])
    prompt_max_sim = non_one_max
    prompt_min_sim = np.min(prompt_similarities)
    new_similarities = np.clip(prompt_similarities, prompt_min_sim, prompt_max_sim)
    normalized_similarities = (new_similarities - prompt_min_sim) / (prompt_max_sim - prompt_min_sim)
    return normalized_similarities


MODEL_ATTRIBUTES = {}
ADDITIONAL_POINTS_CACHE = {}
MODEL_NAME_MAPPING = {
    "small": SMALL_MODEL,
    "large": LARGER_MODEL
}
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    print("Loading data")
    global MODEL_ATTRIBUTES
    for name, model in [("small", SMALL_MODEL), ("large", LARGER_MODEL)]:
        MODEL_ATTRIBUTES[name] = load_data(model)
    print("Data loaded")


class Prompt(BaseModel):
    text: Optional[str] = Field(default=PROMPT)
    
class MostSimilarGlobalRequest(BaseModel):
    token_idx: int
    layer_idx: Optional[int] = Field(default=0)
    prompt: Optional[Prompt]
    num_tokens: Optional[int] = Field(default=100)
    model: Optional[str] = Field(default="small")
    table: Optional[Literal["input", "output"]] = Field(default="input")

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
    model: Optional[str] = Field(default="small")

class TokenSimilaritiesResponse(BaseModel):
    tokens: list[str]
    similarities: list[float]

class CloudRequest(BaseModel):
    sample_rate: float = Field(lt=1, gt = 0, default = 0.1)
    model: Optional[str] = Field(default="small")

class Get2DPointsRequest(BaseModel):
    prompt: Prompt
    layer_idx: Optional[int] = Field(default=0)
    model: Optional[str] = Field(default="small")
class CloudResponse(BaseModel):
    tokens: list[str]
    x: list[float]
    y: list[float]
    total_count: Optional[int] = Field(default=None)

class RootResponse(BaseModel):
    message: str


def tokenize_prompt(model:str, token_ids: list[int]) -> dict[str, list]:
    tokenizer = MODEL_ATTRIBUTES[model].get("tokenizer")
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [token.replace("▁", " ").replace("Ġ", " ") for token in tokens]
    return {"tokens": tokens}


@app.post("/get_tokens_from_ids", response_model=GetTokenResponse)
def get_tokens_from_ids(model:str, token_ids: list[int]) -> GetTokenResponse:
    tokenizer = MODEL_ATTRIBUTES[model].get("tokenizer")
    return tokenizer.convert_ids_to_tokens(token_ids)


@app.post("/get_ids_from_tokens", response_model=GetTokenIdsResponse)
def get_ids_from_tokens(model:str, tokens: list[str]) -> GetTokenIdsResponse:
    tokenizer = MODEL_ATTRIBUTES[model].get("tokenizer")
    return tokenizer.convert_tokens_to_ids(tokens)


@app.post("/token_similarities", response_model=TokenSimilaritiesResponse)
def get_token_similarities(request: TokenSimilaritiesRequest) -> TokenSimilaritiesResponse:
    token_idx = request.token_idx
    layer_idx = request.layer_idx
    model = request.model

    if request.prompt.text:
        prompt = request.prompt.text
    else:
        prompt = PROMPT
    tokenizer = MODEL_ATTRIBUTES[model].get("tokenizer")
    token_ids = get_prompt_token_ids(prompt, tokenizer)
    if layer_idx == 0:
        embeddings_table = MODEL_ATTRIBUTES[model].get("embeddings_table")
        prompt_embeddings = get_token_embeddings(token_ids, embeddings_table)
    else:
        hidden_states = MODEL_ATTRIBUTES[model].get("hidden_states")
        prompt_embeddings = hidden_states.hidden_states[layer_idx].squeeze(0)

    current_emb = prompt_embeddings[token_idx].reshape(1, -1)
    print(f"Comparing token {token_idx} in layer {layer_idx} with {len(prompt_embeddings)} tokens")
    similarities = cosine_similarity_except_self(current_emb, prompt_embeddings).flatten().tolist()
    tokens = tokenize_prompt(model, tokenizer.encode(prompt, add_special_tokens=False))["tokens"]

    return TokenSimilaritiesResponse(
        tokens=tokens,
        similarities=similarities
    )


@app.post("/get_most_similar_global", response_model=MostSimilarGlobalResponse)
def get_most_similar_global(request: MostSimilarGlobalRequest) -> MostSimilarGlobalResponse:
    token_idx = request.token_idx
    layer_idx = request.layer_idx
    num_tokens = request.num_tokens
    model = request.model
    table = request.table

    if request.prompt.text:
        prompt = request.prompt.text
    else:
        prompt = PROMPT

    tokenizer = MODEL_ATTRIBUTES[model].get("tokenizer")
    token_ids = get_prompt_token_ids(prompt, tokenizer)
    
    if table == "input":
        embeddings_table = MODEL_ATTRIBUTES[model].get("input_embeddings_table")
    elif table == "output":
        embeddings_table = MODEL_ATTRIBUTES[model].get("output_embeddings_table")
    else:
        raise HTTPException(status_code=400, detail="Table not found")

    print(f"embeddings_table shape: {embeddings_table.shape}")
    if layer_idx == 0:
        prompt_embeddings = get_token_embeddings(token_ids, embeddings_table)
    else:
        hidden_states = MODEL_ATTRIBUTES[model].get("hidden_states")
        prompt_embeddings = hidden_states.hidden_states[layer_idx].squeeze(0)
    print(f"prompt_embeddings shape: {prompt_embeddings.shape}")
    current_emb = prompt_embeddings[token_idx].reshape(1, -1)

    # Calculate similarities with all tokens in the vocabulary
    print(f"current_emb shape: {current_emb.shape}")
    print(f"embeddings_table shape: {embeddings_table.shape}")
    all_similarities = cosine_similarity(current_emb, embeddings_table).flatten()
    print(f"all_similarities shape: {all_similarities.shape}")
    # Get the top 50 most similar tokens
    top_k = num_tokens
    sort_idx = np.argsort(all_similarities)[::-1]
    top_indices = sort_idx[1:top_k+1] # exclude the current token
    top_similarities = all_similarities[top_indices]
    tokens = tokenizer.batch_decode(top_indices)

    return MostSimilarGlobalResponse(
        tokens=tokens,
        similarities=top_similarities
    )


# def get_2d_cloud_points(model:str, sample_rate: float):
#     embeddings_table = MODEL_ATTRIBUTES[model].get("embeddings_table")
#     umap_model = MODEL_ATTRIBUTES[model].get("umap_model")
#     model_name = MODEL_NAME_MAPPING.get(model, None)
#     if not model_name:
#         raise HTTPException(status_code=400, detail="Model not found")
#     points = get_2d_representation(model_name, umap_model, embeddings_table, load=True)

#     # Match tokens to points
#     tokens = MODEL_ATTRIBUTES[model].get("tokenizer").convert_ids_to_tokens(np.arange(len(points)))
#     mask = [token is not None for token in tokens]
#     # Set sample rate of mask to True
#     sampling_mask = np.random.choice([True, False], size=len(tokens), p=[sample_rate, 1-sample_rate])
#     # Combine the masks
#     mask = np.logical_and(mask, sampling_mask)

#     filtered_tokens = list(compress(tokens, mask))
#     filtered_x = points[:, 0][mask].tolist()
#     filtered_y = points[:, 1][mask].tolist()

#     total_count = len(filtered_tokens)
#     return dict(
#         tokens=filtered_tokens,
#         x=filtered_x,
#         y=filtered_y,
#         total_count=total_count
#     )


# @app.post("/get_2d_cloud", response_model=CloudResponse)
# def get_2d_cloud(request: CloudRequest) -> CloudResponse:
#     sample_rate = request.sample_rate
#     model = request.model
#     data = get_2d_cloud_points(model, sample_rate)
#     return CloudResponse(**data)


# @app.post("/get_additional_points", response_model=CloudResponse)
# def get_additional_points(request: Get2DPointsRequest) -> CloudResponse:
#     layer_idx = request.layer_idx
#     model = request.model

#     if request.prompt.text:
#         prompt = request.prompt.text
#     else:
#         prompt = PROMPT

#     if ADDITIONAL_POINTS_CACHE.get((model, prompt, layer_idx), None):
#         return CloudResponse(**ADDITIONAL_POINTS_CACHE.get((model, prompt, layer_idx)))

#     tokenizer = MODEL_ATTRIBUTES[model].get("tokenizer")
#     token_ids = get_prompt_token_ids(prompt, tokenizer)

#     if layer_idx == 0:
#         embeddings_table = MODEL_ATTRIBUTES[model].get("embeddings_table")
#         prompt_embeddings = get_token_embeddings(token_ids, embeddings_table)
#     else:
#         hidden_states = MODEL_ATTRIBUTES[model].get("hidden_states")
#         prompt_embeddings = hidden_states.hidden_states[layer_idx].squeeze(0)
    
#     umap_model = MODEL_ATTRIBUTES[model].get("umap_model")
#     points = umap_model.transform(prompt_embeddings)
#     tokens = tokenize_prompt(model, tokenizer.encode(prompt, add_special_tokens=False))["tokens"]

#     ADDITIONAL_POINTS_CACHE[(model, prompt, layer_idx)] = dict(
#         tokens=tokens,
#         x=points[:, 0],
#         y=points[:, 1],
#         total_count=len(tokens)
#     )

#     return CloudResponse(**ADDITIONAL_POINTS_CACHE[(model, prompt, layer_idx)])



@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(message="Welcome to the Phi-3 Transformer Visualization API")

@app.on_event("shutdown")
async def shutdown_event():
    print("Application is shutting down")

