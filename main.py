# %%
import pandas as pd
import numpy as np
import config
from data_processing.loader import load_data, load_embeddings
from generation.prompt_generator import (
    generate_prompt_baseline,
    generate_prompt_fewshot,
    generate_prompt_rag,
)
from generation.answers_generator import generate_answers
from postprocess import prepare_submission_file
from rag.retrieval import build_faiss_index, embed_text, retrieve_articles
from transformers import AutoModel, AutoTokenizer
import json

# %%
# Configure the generative AI model
client = config.get_client()

# Model Selection
BASELINE_MODEL = "google/gemini-2.5-pro"
FEWSHOT_MODEL = "google/gemini-2.0-flash"
RAG_MODEL = "google/gemini-2.5-pro"
SUMMARIZATION_MODEL = "google/gemini-2.5-pro"

# Load data
result = load_data(config.XML_FILE)
ground_truth = load_data(config.JSON_KEY_FILE)

# %% [markdown]
# ### Baseline experiment
# %%
baseline_answers = generate_answers(
    cases=result["cases"],
    client=client,
    model_name=BASELINE_MODEL,
    prompt_fn=generate_prompt_baseline
)
prepare_submission_file(
    baseline_answers,
    f"{config.OUTPUT_DIR}/submission_baseline.json",
    result,
    client,
    SUMMARIZATION_MODEL
)

# %% [markdown]
# ### Few-shot experiments
# %%
df = pd.read_pickle(config.EXEMPLARS_FILE)
all_cases = df.to_dict(orient="records")
with open('prompts/example.json', 'r') as f:
    basic_examples = json.load(f)

# %% [markdown]
# #### Few-shot: basic
# %%
fewshot1_answers = generate_answers(
    cases=result["cases"],
    client=client,
    model_name=FEWSHOT_MODEL,
    prompt_fn=lambda case: generate_prompt_fewshot(case, basic_examples),
    max_retries=5
)
prepare_submission_file(
    fewshot1_answers,
    f"{config.OUTPUT_DIR}/submission_fewshot1.json",
    result,
    client,
    SUMMARIZATION_MODEL
)

# %% [markdown]
# #### Few-shot: LLM-generated exemplars
# %%
fewshot2_answers = generate_answers(
    cases=result["cases"],
    client=client,
    model_name=FEWSHOT_MODEL,
    prompt_fn=lambda case: generate_prompt_fewshot(case, all_cases),
    max_retries=5
)
prepare_submission_file(
    fewshot2_answers,
    f"{config.OUTPUT_DIR}/submission_fewshot2.json",
    result,
    client,
    SUMMARIZATION_MODEL
)

# %% [markdown]
# #### Few-shot: exemplars with reasoning
# %%
fewshot3_answers = generate_answers(
    cases=result["cases"],
    client=client,
    model_name=FEWSHOT_MODEL,
    prompt_fn=lambda case: generate_prompt_fewshot(case, all_cases, add_reasoning=True),
    max_retries=5
)
prepare_submission_file(
    fewshot3_answers,
    f"{config.OUTPUT_DIR}/submission_fewshot3.json",
    result,
    client,
    SUMMARIZATION_MODEL
)

# %% [markdown]
# ### RAG experiments
# %%
df_article = pd.read_csv(config.ARTICLE_FILE)
df_query = pd.read_csv(config.QUERY_FILE)

# Load model for RAG
query_model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)
query_tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)

# Create embeddings and build index
question_embeddings = embed_text(df_query['Clinician Question'].tolist(), query_model, query_tokenizer)
article_embeddings = load_embeddings(config.ARTICLE_FILE)
index = build_faiss_index(article_embeddings)

# Retrieve articles for all queries
retrieved_articles = retrieve_articles(np.array(question_embeddings), index, df_article)

# %% [markdown]
# #### RAG: full articles
# %%
rag1_answers = generate_answers(
    cases=result["cases"],
    client=client,
    model_name=RAG_MODEL,
    prompt_fn=lambda case, idx: generate_prompt_rag(case, retrieved_articles[idx]),
    max_retries=5,
    use_index=True,
)
prepare_submission_file(
    rag1_answers,
    f"{config.OUTPUT_DIR}/submission_gemini_rag1.json",
    result,
    client,
    SUMMARIZATION_MODEL
)


