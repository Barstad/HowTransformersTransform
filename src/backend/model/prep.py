from model.utils import (
    get_2d_representation, load_model, create_tokenizer, create_hidden_states,
    create_token_embeddings_table, create_umap_model, PROMPT
)

def preprocess_and_save(model_name:str):
    print(f"Loading model and tokenizer for {model_name}...")
    model = load_model(model_name)
    tokenizer = create_tokenizer(model_name)

    print(f"Saving data for {model_name}...")
    create_hidden_states(model, model_name, tokenizer, PROMPT)
    embeddings = create_token_embeddings_table(model_name, model)
    umap_model = create_umap_model(model_name, embeddings)
    get_2d_representation(model_name, umap_model, embeddings, save=True)
    print(f"Saved data for {model_name} to 'data' directory.")
    del model
    del tokenizer
    del embeddings
    
def prep_models(models:list[str]):
    for model in models:
        preprocess_and_save(model)

if __name__ == "__main__":
    from model.utils import SMALL_MODEL, LARGER_MODEL
    prep_models([SMALL_MODEL, LARGER_MODEL])

