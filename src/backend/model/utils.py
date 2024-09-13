import pickle
import os
from typing import Any, Callable, Literal
import umap
import os
import numpy as np
from pathlib import Path




PROMPT = Path("prompt.txt").read_text()
SMALL_MODEL = "microsoft/Phi-3.5-mini-instruct"
LARGER_MODEL = "Qwen/Qwen2-7B"

def get_tokenizer_path(model_name):
    path = Path(f"data/{model_name.split('/')[-1]}/tokenizer.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_hidden_states_path(model_name):
    path = Path(f"data/{model_name.split('/')[-1]}/prompt_hidden_states.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_token_embeddings_path(model_name, variant: str):
    path = Path(f"data/{model_name.split('/')[-1]}/{variant}_token_embeddings.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_umap_path(model_name):
    path = Path(f"data/{model_name.split('/')[-1]}/umap.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_2d_representation_path(model_name):
    path = Path(f"data/{model_name.split('/')[-1]}/emb_2d.npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def load_model(model_name:str):
    print("Loading model...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    print("Model loaded successfully.")
    return model


def create_tokenizer(model_name:str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    path = get_tokenizer_path(model_name)
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    return tokenizer


def get_tokenizer(model:str):
    path = get_tokenizer_path(model)
    if not path.exists():
        raise ValueError(f"Tokenizer for model {model} not found at {path}")
    with open(path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def get_prompt_token_ids(prompt:str, tokenizer:Any):
    assert prompt == PROMPT, "Only supporting one prompt"
    token_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    return token_ids


def create_hidden_states(model:Callable, model_name:str, tokenizer:Any, prompt:str=None):
    if prompt is None:
        prompt = PROMPT
    assert prompt == PROMPT, "Only supporting one prompt"
    print(f"Creating hidden states for with model {model_name} and prompt {prompt[:30]}...")
    path = get_hidden_states_path(model_name)
    import torch
    token_ids = get_prompt_token_ids(prompt, tokenizer)
    with torch.inference_mode():
        out = model(token_ids, output_hidden_states=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print("Hidden states calculated and saved.")
    return out


def get_hidden_states(model_name:str):
    path = get_hidden_states_path(model_name)
    if not path.exists():
        raise ValueError(f"Hidden states for model {model_name} not found at {path}")
    print("Loading cached output")
    with open(path, "rb") as f:
        out = pickle.load(f)
    return out


def create_token_embeddings_table(model_name:str, model:Callable, variant:Literal["input", "output"]) -> np.ndarray:
    path = get_token_embeddings_path(model_name, variant)
    if variant == "input":
        embeddings = model.model.embed_tokens.weight.detach().numpy()
    elif variant == "output":
        embeddings = model.lm_head.weight.detach().numpy()
    else:
        raise ValueError(f"Invalid variant {variant}")
    np.save(path, embeddings)
    print(f"{variant} token embeddings table calculated and saved.")
    return embeddings

def get_token_embeddings_table(model_name:str, variant:Literal["input", "output"]):
    path = get_token_embeddings_path(model_name, variant)
    if not path.exists():
        raise ValueError(f"{variant} token embeddings table for model {model_name} not found at {path}")
    embeddings = np.load(path)
    return embeddings


def get_token_embeddings(token_ids:Any, embeddings_table:np.ndarray):
    embeddings = embeddings_table[token_ids.flatten()]
    return embeddings


def create_umap_model(model_name:str, input_embeddings_table:np.ndarray, output_embeddings_table:np.ndarray):
    path = get_umap_path(model_name)
    embeddings = np.concatenate([input_embeddings_table, output_embeddings_table])
    umap_model = umap.UMAP(n_neighbors=20, metric="cosine").fit(embeddings)
    with open(path, "wb") as f:
        pickle.dump(umap_model, f)
    return umap_model


def get_umap_model(model_name:str):
    path = get_umap_path(model_name)
    if not path.exists():
        raise ValueError(f"UMAP model for model {model_name} not found at {path}")
    with open(path, "rb") as f:
        umap_model = pickle.load(f)
    return umap_model


def get_2d_representation(model_name:str, umap_model:Callable, embeddings:np.ndarray, save:bool=False, load:bool=False):
    assert not (save and load), "Cannot save and load at the same time"
    path = get_2d_representation_path(model_name)
    if load:
        representation = np.load(path)
    else:
        representation = umap_model.transform(embeddings)
    if save:
        np.save(path, representation)
    return representation


def cosine_similarity(a, b):
    # Ensure a and b are 2D arrays
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    
    # # Compute norms
    # norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    # norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    
    # # Handle zero norms
    # epsilon = 1e-8  # Small value to avoid division by zero
    # norm_a = np.maximum(norm_a, epsilon)
    # norm_b = np.maximum(norm_b, epsilon)
    
    # Compute similarity
    # similarity = np.dot(a, b.T) / (norm_a * norm_b.T)
    similarity = np.dot(a, b.T)

    return similarity.astype(np.float32)

def get_top_similar_tokens(
        token_embedding,
        all_token_embeddings,
        n=5,
        decode=False,
        return_similarity=False,
        tokenizer=None):
    cosine_sim = cosine_similarity(token_embedding, all_token_embeddings).flatten()
    top_indices = np.argsort(cosine_sim)[-n:][::-1]
    if decode:
        if tokenizer is None:
            raise ValueError("Tokenizer is required for decoding")
        result = [tokenizer.decode([i]) for i in top_indices]
    else:
        result = top_indices
    if return_similarity:
        return result, cosine_sim[top_indices]
    return result


