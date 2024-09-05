import pickle
import os
import umap
import os
import numpy as np



PROMPT = """An android named Apple was lost in the woods. Luckily, he had a phone with Google Maps. As he pulled out the device, its screen flickered to life, casting a faint blue glow on his metallic face. Apple's optical sensors scanned the map, processing the terrain data.
"Calculating optimal route," he muttered, his voice a melodic hum of circuits and synthesized speech.
Suddenly, a drop of water splashed onto the screen. Then another. Apple looked up, raindrops now pattering against his waterproof casing. The weather was interfering with his GPS signal.
As the downpour intensified, Apple realized he faced a new challenge. His circuitry was protected, but the phone wasn't waterproof. He needed shelter, and fast."""

def load_model():
    print("Loading model...")
    from phi35.simple_phi3 import Phi3Model
    model = Phi3Model.from_pretrained("microsoft/Phi-3.5-mini-instruct", )
    print("Model loaded successfully.")
    return model

def load_tokenizer():
    
    path = "data/tokenizer.pkl"
    if not os.path.exists(path):
        print("Creating new tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        pickle.dump(tokenizer, open(path, "wb"))
    else:
        print("Loading tokenizer...")
        tokenizer = pickle.load(open(path, "rb"))
    print("Tokenizer loaded successfully.")
    return tokenizer

def get_prompt_token_ids(prompt:str, tokenizer):
    print(f"Tokenizing prompt: '{prompt[:30]}...'")
    token_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    print(f"Prompt tokenized. Number of tokens: {token_ids.shape[1]}")
    return token_ids

def get_hidden_states(prompt, tokenizer, model=None, recalculate=False):
    print(f"Getting hidden states. Recalculate: {recalculate}")
    path = "data/out.pkl"   
    if not recalculate and os.path.exists(path):
        print("Loading cached output")
        out = pickle.load(open(path, "rb"))
    else:
        import torch
        print("Recalculating hidden states...")
        if model is None:
            model = load_model()
        model.eval()
        token_ids = get_prompt_token_ids(prompt, tokenizer)
        with torch.inference_mode():
            out = model(token_ids, output_hidden_states=True)
        pickle.dump(out, open(path, "wb"))
        print("Hidden states calculated and saved.")
    return out

def get_token_embeddings_table(model=None, recalculate=False):
    print("Retrieving token embeddings table...")
    path = "data/token_embeddings.npy"
    if not recalculate and os.path.exists(path):
        print("Loading existing token embeddings table")
        embeddings = np.load(path)
    else:
        print("Calculating token embeddings table...")
        if model is None:
            model = load_model()
        embeddings = model.embed_tokens.weight.detach().numpy()
        np.save(path, embeddings)
        print("Token embeddings table calculated and saved.")

    print(f"Token embeddings table shape: {embeddings.shape}")
    return embeddings


def get_token_embeddings(token_ids, embeddings_table):
    print("Getting original embeddings for tokens...")
    embeddings = embeddings_table[token_ids.flatten()]
    print(f"Original embeddings shape: {embeddings.shape}")
    return embeddings


def get_umap_model(embeddings_table = None, model = None, recalculate=False):
    print(f"Getting UMAP model. Recalculate: {recalculate}")
    path = "data/umap.pkl"
    if not recalculate and os.path.exists(path):
        print("Loading existing UMAP model")
        umap_model = pickle.load(open(path, "rb"))
    else:
        if embeddings_table is None:
            embeddings_table = get_token_embeddings_table(model)
        print("Creating new UMAP model...")
        umap_model = umap.UMAP(n_neighbors=20, metric="cosine").fit(embeddings_table)
        pickle.dump(umap_model, open(path, "wb"))
        print("UMAP model created and saved.")
    return umap_model


def get_2d_representation(embeddings, umap_model, recalculate=False):
    path = "data/emb_2d.npy"
    if not recalculate and os.path.exists(path):
        print("Loading existing 2D representation")
        representation = np.load(path)
    else:
        print("Transforming embeddings to 2D representation...")
        representation = umap_model.transform(embeddings)
        np.save(path, representation)
    print(f"2D representation shape: {representation.shape}")
    return representation


def plot_2d_token_representation(emb_2d, tokenizer):
    from bokeh.plotting import figure, show
    from bokeh.io import output_notebook
    from bokeh.models import ColumnDataSource, HoverTool

    # Enable Bokeh to work in Jupyter notebook
    output_notebook()

    # Create a ColumnDataSource for the data
    source = ColumnDataSource(data=dict(
        x=emb_2d[:, 0],
        y=emb_2d[:, 1],
        token=[tokenizer.decode([i]) for i in range(len(emb_2d))]
    ))

    # Create the figure
    p = figure(width=800, height=600, title="Token Embeddings Visualization")

    # Add the scatter plot
    p.scatter('x', 'y', source=source, alpha=0.6)

    # Add hover tool
    hover = HoverTool(tooltips=[
        ("Token", "@token"),
        ("x", "@x{0.000}"),
        ("y", "@y{0.000}")
    ])
    p.add_tools(hover)

    # Show the plot
    show(p)

def cosine_similarity(a, b):
    print("Calculating cosine similarity...")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    similarity = np.dot(a, b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1))
    print(f"Cosine similarity shape: {similarity.shape}")
    return similarity

def get_top_similar_tokens(
        token_embedding,
        all_token_embeddings,
        n=5,
        decode=False,
        tokenizer=None):
    print(f"Getting top {n} similar tokens...")
    cosine_sim = cosine_similarity(token_embedding, all_token_embeddings).flatten()
    top_indices = np.argsort(cosine_sim)[-n:][::-1]
    if decode:
        if tokenizer is None:
            raise ValueError("Tokenizer is required for decoding")
        result = [tokenizer.decode([i]) for i in top_indices]
    else:
        result = top_indices
    print(f"Top similar tokens: {result}")
    return result


