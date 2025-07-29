import click
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

@click.group()
def cli():
    """A CLI to run text generation experiments."""
    pass

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.5-pro', help='Model to use for the baseline experiment.')
@click.option('--summarizer', 'summarizer_model', default='google/gemini-2.5-pro', help='Model to use for summarization.')
def baseline(model_name, summarizer_model):
    """Runs the baseline experiment."""
    client = config.get_client()
    result = load_data(config.XML_FILE)
    
    click.echo(f"Running baseline experiment with model: {model_name}")
    baseline_answers = generate_answers(
        cases=result["cases"],
        client=client,
        model_name=model_name,
        prompt_fn=generate_prompt_baseline
    )
    prepare_submission_file(
        baseline_answers,
        f"{config.OUTPUT_DIR}/submission_baseline.json",
        result,
        client,
        summarizer_model
    )
    click.echo("Baseline experiment complete.")

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.0-flash', help='Model to use for the few-shot experiments.')
@click.option('--summarizer', 'summarizer_model', default='google/gemini-2.5-pro', help='Model to use for summarization.')
def few_shot(model_name, summarizer_model):
    """Runs the few-shot experiments."""
    client = config.get_client()
    result = load_data(config.XML_FILE)
    
    df = pd.read_pickle(config.EXEMPLARS_FILE)
    all_cases = df.to_dict(orient="records")
    with open('prompts/example.json', 'r') as f:
        basic_examples = json.load(f)

    click.echo(f"Running few-shot (basic) experiment with model: {model_name}")
    fewshot1_answers = generate_answers(
        cases=result["cases"],
        client=client,
        model_name=model_name,
        prompt_fn=lambda case: generate_prompt_fewshot(case, basic_examples),
        max_retries=5
    )
    prepare_submission_file(
        fewshot1_answers,
        f"{config.OUTPUT_DIR}/submission_fewshot1.json",
        result,
        client,
        summarizer_model
    )
    
    click.echo(f"Running few-shot (LLM-generated exemplars) experiment with model: {model_name}")
    fewshot2_answers = generate_answers(
        cases=result["cases"],
        client=client,
        model_name=model_name,
        prompt_fn=lambda case: generate_prompt_fewshot(case, all_cases),
        max_retries=5
    )
    prepare_submission_file(
        fewshot2_answers,
        f"{config.OUTPUT_DIR}/submission_fewshot2.json",
        result,
        client,
        summarizer_model
    )

    click.echo(f"Running few-shot (exemplars with reasoning) experiment with model: {model_name}")
    fewshot3_answers = generate_answers(
        cases=result["cases"],
        client=client,
        model_name=model_name,
        prompt_fn=lambda case: generate_prompt_fewshot(case, all_cases, add_reasoning=True),
        max_retries=5
    )
    prepare_submission_file(
        fewshot3_answers,
        f"{config.OUTPUT_DIR}/submission_fewshot3.json",
        result,
        client,
        summarizer_model
    )
    click.echo("Few-shot experiments complete.")

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.5-pro', help='Model to use for the RAG experiment.')
@click.option('--summarizer', 'summarizer_model', default='google/gemini-2.5-pro', help='Model to use for summarization.')
def rag(model_name, summarizer_model):
    """Runs the RAG experiment."""
    client = config.get_client()
    result = load_data(config.XML_FILE)

    df_article = pd.read_csv(config.ARTICLE_FILE)
    df_query = pd.read_csv(config.QUERY_FILE)
    
    query_model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)
    query_tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
    
    question_embeddings = embed_text(df_query['Clinician Question'].tolist(), query_model, query_tokenizer)
    article_embeddings = load_embeddings(config.ARTICLE_FILE)
    index = build_faiss_index(article_embeddings)
    
    retrieved_articles = retrieve_articles(np.array(question_embeddings), index, df_article)

    click.echo(f"Running RAG experiment with model: {model_name}")
    rag1_answers = generate_answers(
        cases=result["cases"],
        client=client,
        model_name=model_name,
        prompt_fn=lambda case, idx: generate_prompt_rag(case, retrieved_articles[idx]),
        max_retries=5,
        use_index=True,
    )
    prepare_submission_file(
        rag1_answers,
        f"{config.OUTPUT_DIR}/submission_gemini_rag1.json",
        result,
        client,
        summarizer_model
    )
    click.echo("RAG experiment complete.")

@cli.command()
@click.pass_context
def all(ctx):
    """Runs all experiments with their default models."""
    ctx.invoke(baseline)
    ctx.invoke(few_shot)
    ctx.invoke(rag)
    click.echo("All experiments complete.")

if __name__ == '__main__':
    cli() 