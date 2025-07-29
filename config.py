import google.generativeai as genai

GENAI_API_KEY = "<YOUR_API_KEY>"

DATA_DIR = "data"
OUTPUT_DIR = "output"
XML_FILE = f"{DATA_DIR}/archehr-qa.xml"
JSON_KEY_FILE = f"{DATA_DIR}/archehr-qa_key.json"
EXEMPLARS_FILE = f"{DATA_DIR}/syn_answer.pkl"
ARTICLE_FILE = f"{DATA_DIR}/mds_manual_data_with_embeddings.csv"
QUERY_FILE = f"{DATA_DIR}/query_data.csv"

def configure_genai():
    genai.configure(api_key=GENAI_API_KEY)

def get_model(model_name="gemini-1.5-flash"):
    return genai.GenerativeModel(model_name=model_name) 