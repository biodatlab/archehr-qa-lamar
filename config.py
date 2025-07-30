import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

EMBEDDING_MODEL = "ncbi/MedCPT-Query-Encoder"

DATA_DIR = "data/"
OUTPUT_DIR = "output/"
XML_FILE = f"{DATA_DIR}/archehr-qa.xml"
JSON_KEY_FILE = f"{DATA_DIR}/archehr-qa_key.json"
SYNTHETIC_ANSWERS_FILE = f"{DATA_DIR}/syn_answer.pkl"
SYNTHETIC_ANSWERS_WITH_REASONING_FILE = f"{DATA_DIR}/syn_answer_with_reasoning.pkl"
SUMMARIZED_ARTICLES_FILE = f"{DATA_DIR}/summarized_articles.pkl"
SYNTHETIC_CASES_FILE = f"{DATA_DIR}/synthetic_cases.csv"
ARTICLE_FILE = f"{DATA_DIR}/mds_manual_data_with_embeddings.csv"
QUERY_FILE = f"{DATA_DIR}/query_data.csv"

def get_client():
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL) 