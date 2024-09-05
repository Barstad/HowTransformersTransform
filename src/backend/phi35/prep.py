from phi35.utils import (
    load_model, load_tokenizer, get_hidden_states,
    get_token_embeddings_table, get_umap_model, PROMPT
)

def preprocess_and_save():
    print("Loading model and tokenizer...")
    model = load_model()
    tokenizer = load_tokenizer()

    print("Saving data...")
    get_hidden_states(PROMPT, model, tokenizer)
    embeddings = get_token_embeddings_table(model)
    get_umap_model(embeddings)
    print("Saved data to 'data' directory.")