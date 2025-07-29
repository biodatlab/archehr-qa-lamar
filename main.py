# %%
import pandas as pd
import numpy as np
import config
from data_processing.loader import load_data
from generation.prompt_generator import (
    generate_prompt_baseline,
    generate_prompt_fewshot,
    generate_prompt_rag,
    example1,
    example2
)
from generation.answers_generator import generate_answers
from postprocess import prepare_submission_file
from rag.retrieval import build_faiss_index, embed_text, retrieve_articles
from transformers import AutoModel, AutoTokenizer

# %%
# Configure the generative AI model
config.configure_genai()
model = config.get_model()

# Load data
result = load_data(config.XML_FILE)
ground_truth = load_data(config.JSON_KEY_FILE)

# %% [markdown]
# ### Baseline experiment
# %%
generate_answers(
    cases=result["cases"],
    model=model,
    prompt_fn=generate_prompt_baseline,
    answers_json_path=f"{config.OUTPUT_DIR}/answer_baseline.json"
)
prepare_submission_file(
    f"{config.OUTPUT_DIR}/answer_baseline.json",
    f"{config.OUTPUT_DIR}/submission_baseline.json",
    result,
    model
)

# %% [markdown]
# ### Few-shot experiments
# %%
df = pd.read_pickle(config.EXEMPLARS_FILE)
all_cases = df.to_dict(orient="records")

# %% [markdown]
# #### Few-shot: basic
# %%
generate_answers(
    cases=result["cases"],
    model=model,
    prompt_fn=lambda case: generate_prompt_fewshot(case, [example1, example2]),
    answers_json_path=f"{config.OUTPUT_DIR}/answer_fewshot1.json",
    max_retries=5
)
prepare_submission_file(
    f"{config.OUTPUT_DIR}/answer_fewshot1.json",
    f"{config.OUTPUT_DIR}/submission_fewshot1.json",
    result,
    model
)

# %% [markdown]
# #### Few-shot: LLM-generated exemplars
# %%
generate_answers(
    cases=result["cases"],
    model=model,
    prompt_fn=lambda case: generate_prompt_fewshot(case, all_cases),
    answers_json_path=f"{config.OUTPUT_DIR}/answer_fewshot2.json",
    max_retries=5
)
prepare_submission_file(
    f"{config.OUTPUT_DIR}/answer_fewshot2.json",
    f"{config.OUTPUT_DIR}/submission_fewshot2.json",
    result,
    model
)

# %% [markdown]
# #### Few-shot: exemplars with reasoning
# %%
generate_answers(
    cases=result["cases"],
    model=model,
    prompt_fn=lambda case: generate_prompt_fewshot(case, all_cases, add_reasoning=True),
    answers_json_path=f"{config.OUTPUT_DIR}/answer_fewshot3.json",
    max_retries=5
)
prepare_submission_file(
    f"{config.OUTPUT_DIR}/answer_fewshot3.json",
    f"{config.OUTPUT_DIR}/submission_fewshot3.json",
    result,
    model
)

# %% [markdown]
# ### RAG experiments
# %%
df_article = pd.read_csv(config.ARTICLE_FILE)
df_query = pd.read_csv(config.QUERY_FILE)

# Load model for RAG
query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

# Create embeddings and build index
question_embeddings = embed_text(df_query['Clinician Question'].tolist(), query_model, query_tokenizer)
article_embeddings = np.array(df_article['embeddings'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ')).tolist())
index = build_faiss_index(article_embeddings)

# Retrieve articles for all queries
retrieved_articles = retrieve_articles(np.array(question_embeddings), index, df_article)

# %% [markdown]
# #### RAG: full articles
# %%
generate_answers(
    cases=result["cases"],
    model=model,
    prompt_fn=lambda case, idx: generate_prompt_rag(case, retrieved_articles[idx]),
    answers_json_path=f"{config.OUTPUT_DIR}/answer_gemini_rag1.json",
    max_retries=5,
    use_index=True,
)
prepare_submission_file(
    f"{config.OUTPUT_DIR}/answer_gemini_rag1.json",
    f"{config.OUTPUT_DIR}/submission_gemini_rag1.json",
    result,
    model
)


