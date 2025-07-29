import faiss
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm.auto import tqdm

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def embed_text(texts, model, tokenizer, batch_size=10):
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

def retrieve_articles(query_embeddings, index, df_article, k=5):
    distances, indices = index.search(query_embeddings, k)
    retrieved_articles_per_query = []
    for i in range(query_embeddings.shape[0]):
        retrieved_articles_for_query = []
        for j in range(k):
            article_index = indices[i, j]
            retrieved_article = df_article.iloc[article_index]
            retrieved_articles_for_query.append({
                "title": retrieved_article["title"],
                "article": retrieved_article["markdown"],
            })
        retrieved_articles_per_query.append(retrieved_articles_for_query)
    return retrieved_articles_per_query 