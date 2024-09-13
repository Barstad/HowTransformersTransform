import json
import numpy as np
from model.utils import (
    LARGER_MODEL,
    SMALL_MODEL,
    cosine_similarity,
    get_hidden_states,
    get_prompt_token_ids,
    get_token_embeddings,
    get_token_embeddings_table,
    get_tokenizer,
    load_model,
    create_tokenizer,
    create_hidden_states,
    create_token_embeddings_table,
    PROMPT
)


MODEL_NAME_MAPPING = {
    "small": SMALL_MODEL,
    "large": LARGER_MODEL
}



def preprocess_and_save(model_name:str):
    print(f"Loading model and tokenizer for {model_name}...")
    model = load_model(model_name)
    tokenizer = create_tokenizer(model_name)

    print(f"Saving data for {model_name}...")
    create_hidden_states(model, model_name, tokenizer, PROMPT)
    create_token_embeddings_table(model_name, model, "input")
    create_token_embeddings_table(model_name, model, "output")
    # umap_model = create_umap_model(model_name, input_embeddings, output_embeddings)
    # get_2d_representation(model_name, umap_model, embeddings, save=True)
    print(f"Saved data for {model_name} to 'data' directory.")
    del model
    del tokenizer


def prep_models(models:list[str]):
    for model in models:
        preprocess_and_save(model)



def similarity_except_max(emb, embeddings):
    prompt_similarities = cosine_similarity(emb, embeddings)
    max_sim = np.max(prompt_similarities, axis = 1)
    mask = prompt_similarities<max_sim

    second_to_max = np.max(np.where(~mask, -1e9, prompt_similarities), axis = 1)
    clipped = np.clip(prompt_similarities, np.min(prompt_similarities, axis = 1), second_to_max)

    normalized = (clipped - np.min(prompt_similarities, axis = 1)) / (second_to_max - np.min(prompt_similarities, axis = 1))
    return normalized


def load_data(model_name:str) -> tuple:
    tokenizer = get_tokenizer(model_name)
    hidden_states = get_hidden_states(model_name)
    input_embeddings_table = get_token_embeddings_table(model_name, "input")
    output_embeddings_table = get_token_embeddings_table(model_name, "output")
    token_ids = get_prompt_token_ids(PROMPT, tokenizer)
    return tokenizer, hidden_states, input_embeddings_table, output_embeddings_table, token_ids


def load_model_attributes() -> dict:
    model_attributes = {}
    for model_abbr, model_name in MODEL_NAME_MAPPING.items():
        tokenizer, hidden_states, input_embeddings_table, output_embeddings_table, token_ids = load_data(model_name)
        model_attributes[model_abbr] = {
            "tokenizer": tokenizer,
            "hidden_states": hidden_states,
            "input_embeddings_table": input_embeddings_table,
            "output_embeddings_table": output_embeddings_table,
            "token_ids": token_ids
        }
    return model_attributes


def tokenize_prompt(model_attributes: dict, token_ids: list[int]) -> dict[str, list]:
    tokenizer = model_attributes.get("tokenizer")
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = [token.replace("▁", " ").replace("Ġ", " ") for token in tokens]
    return {"tokens": tokens}


def get_token_similarities(model_attributes: dict, token_idx, layer_idx, prompt = PROMPT) -> dict:
    tokenizer = model_attributes.get("tokenizer")
    token_ids = get_prompt_token_ids(prompt, tokenizer)
    if layer_idx == 0:
        embeddings_table = model_attributes.get("input_embeddings_table")
        prompt_embeddings = get_token_embeddings(token_ids, embeddings_table)
    else:
        hidden_states = model_attributes.get("hidden_states")
        prompt_embeddings = hidden_states.hidden_states[layer_idx-1].squeeze(0)

    current_emb = prompt_embeddings[token_idx].reshape(1, -1)
    similarities = similarity_except_max(current_emb, prompt_embeddings).flatten().tolist()
    tokens = tokenize_prompt(model_attributes, tokenizer.encode(prompt, add_special_tokens=False))["tokens"]

    return {"tokens": tokens, "similarities": similarities}


def get_most_similar_global(model_attributes: dict, token_idx, layer_idx, num_tokens, table, prompt = PROMPT) -> dict:
    tokenizer = model_attributes.get("tokenizer")
    token_ids = get_prompt_token_ids(prompt, tokenizer)
    
    if table == "input":
        embeddings_table = model_attributes.get("input_embeddings_table")
    elif table == "output":
        embeddings_table = model_attributes.get("output_embeddings_table")
    else:
        raise ValueError("Table not found")

    if layer_idx == 0:
        prompt_embeddings = get_token_embeddings(token_ids, embeddings_table)
    else:
        hidden_states = model_attributes.get("hidden_states")
        prompt_embeddings = hidden_states.hidden_states[layer_idx-1].squeeze(0)
    current_emb = prompt_embeddings[token_idx].reshape(1, -1)

    # Calculate similarities with all tokens in the vocabulary
    all_similarities = similarity_except_max(current_emb, embeddings_table).flatten()
    # Get the top 50 most similar tokens
    top_k = num_tokens
    sort_idx = np.argsort(all_similarities)[::-1]
    top_indices = sort_idx[1:top_k+1] # exclude the current token
    top_similarities = all_similarities[top_indices]
    tokens = tokenizer.batch_decode(top_indices)

    return {"tokens": tokens, "similarities": top_similarities.tolist()}


def prep_input_data():
    model_attributes = load_model_attributes()

    input_data = {}

    for model_abbr, model_attributes in model_attributes.items():
        print(f"Processing model: {model_abbr}")
        tokenizer = model_attributes["tokenizer"]
        tokens = tokenizer.encode(PROMPT, add_special_tokens=False)
        prompt_tokens = tokenize_prompt(model_attributes, tokens)["tokens"]
        input_data[model_abbr] = {}
        input_data[model_abbr]["prompt_tokens"] = prompt_tokens
        input_data[model_abbr]["layers"] = {}
        
        for layer_idx in range(len(model_attributes["hidden_states"].hidden_states)+1): # +1 for embedding table
            input_data[model_abbr]["layers"][layer_idx] = {}
            print(f"Processing layer {layer_idx}/{len(model_attributes['hidden_states'].hidden_states)}")
            for token_idx, token in enumerate(tokens):
                token_similarities = get_token_similarities(model_attributes, token_idx, layer_idx, prompt=PROMPT)["similarities"]
                most_similar_global_input = get_most_similar_global(model_attributes, token_idx, layer_idx, 100, "input", prompt=PROMPT)
                most_similar_global_output = get_most_similar_global(model_attributes, token_idx, layer_idx, 100, "output", prompt=PROMPT)
                input_data[model_abbr]["layers"][layer_idx][token_idx] = {
                    "prompt_token_similarities": token_similarities,
                    "most_similar_global_input": most_similar_global_input,
                    "most_similar_global_output": most_similar_global_output
                }
        print(f"Finished processing model: {model_abbr}")

    print("Input data preparation complete")
    with open("data/input_data.json", "w") as f:
        json.dump(input_data, f)
    return input_data



def split_file():
    with open("data/input_data.json", "r") as f:
        data = json.load(f)

    prompt_similarities = {}
    most_similar_global_input = {}
    most_similar_global_output = {}

    for model_abbr, model_data in data.items():
        prompt_tokens = model_data["prompt_tokens"]
        layers = model_data["layers"]
        prompt_similarities[model_abbr] = {"prompt_tokens": prompt_tokens, "layers": {}}
        most_similar_global_input[model_abbr] = {}
        most_similar_global_output[model_abbr] = {}

        for layer_idx in layers.keys():
            prompt_similarities[model_abbr]["layers"][layer_idx] = {}
            most_similar_global_input[model_abbr][layer_idx] = {}
            most_similar_global_output[model_abbr][layer_idx] = {}

            for token_idx, token in enumerate(prompt_tokens):
                layer_data = layers[str(layer_idx)][str(token_idx)]
                prompt_similarities[model_abbr]["layers"][layer_idx][token_idx] = {
                    "similarities": layer_data["prompt_token_similarities"]}
                most_similar_global_input[model_abbr][layer_idx][token_idx] = layer_data["most_similar_global_input"]
                most_similar_global_output[model_abbr][layer_idx][token_idx] = layer_data["most_similar_global_output"]

    with open("data/prompt_similarities.json", "w") as f:
        json.dump(prompt_similarities, f)
    with open("data/most_similar_global_input.json", "w") as f:
        json.dump(most_similar_global_input, f)
    with open("data/most_similar_global_output.json", "w") as f:
        json.dump(most_similar_global_output, f)