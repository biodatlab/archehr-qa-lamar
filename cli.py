import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import click
import pandas as pd
import numpy as np
import config
from data_processing.loader import load_data, load_embeddings
from generation.prompt_generator import (
    generate_prompt_baseline,
    generate_prompt_fewshot,
    generate_prompt_rag,
    generate_prompt_rag_summary,
    generate_prompt_rag_synthetic_cases
)
from generation.answers_generator import generate_answers
from postprocess import prepare_submission_file
import json
from generation.synthetic_data_generator import (
    generate_synthetic_answers,
    generate_reasoning,
    summarize_articles,
    generate_synthetic_cases
)
import pickle

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
@click.option('--model', 'model_name', default='google/gemini-2.0-flash', help='Model to use for the basic few-shot experiment.')
@click.option('--summarizer', 'summarizer_model', default='google/gemini-2.5-pro', help='Model to use for summarization.')
def few_shot_basic(model_name, summarizer_model):
    """Runs the basic few-shot experiment."""
    client = config.get_client()
    result = load_data(config.XML_FILE)
    
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
    click.echo("Basic few-shot experiment complete.")

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.0-flash', help='Model to use for the LLM-generated few-shot experiment.')
@click.option('--summarizer', 'summarizer_model', default='google/gemini-2.5-pro', help='Model to use for summarization.')
@click.pass_context
def few_shot_syn_ans(ctx, model_name, summarizer_model):
    """Runs the few-shot experiment with LLM-generated exemplars."""
    if not os.path.exists(config.SYNTHETIC_ANSWERS_FILE):
        click.echo("Synthetic answers file not found. Generating it now...")
        ctx.invoke(generate_synthetic_answers_cli, output_path=config.SYNTHETIC_ANSWERS_FILE)

    client = config.get_client()
    result = load_data(config.XML_FILE)
    
    df = pd.read_pickle(config.SYNTHETIC_ANSWERS_FILE)
    all_cases = df.to_dict(orient="records")

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
    click.echo("LLM-generated few-shot experiment complete.")

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.0-flash', help='Model to use for the few-shot with reasoning experiment.')
@click.option('--summarizer', 'summarizer_model', default='google/gemini-2.5-pro', help='Model to use for summarization.')
@click.pass_context
def few_shot_syn_w_reasoning(ctx, model_name, summarizer_model):
    """Runs the few-shot experiment with reasoning."""
    if not os.path.exists(config.SYNTHETIC_ANSWERS_WITH_REASONING_FILE):
        click.echo("Synthetic answers with reasoning file not found. Generating it now...")
        if not os.path.exists(config.SYNTHETIC_ANSWERS_FILE):
            ctx.invoke(generate_synthetic_answers_cli, output_path=config.SYNTHETIC_ANSWERS_FILE)
        ctx.invoke(generate_reasoning_cli, input_path=config.SYNTHETIC_ANSWERS_FILE, output_path=config.SYNTHETIC_ANSWERS_WITH_REASONING_FILE)

    client = config.get_client()
    result = load_data(config.XML_FILE)
    
    df = pd.read_pickle(config.SYNTHETIC_ANSWERS_WITH_REASONING_FILE)
    all_cases = df.to_dict(orient="records")

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
    click.echo("Few-shot with reasoning experiment complete.")

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.5-pro', help='Model to use for generating synthetic answers.')
@click.option('--output', 'output_path', default=config.SYNTHETIC_ANSWERS_FILE, help='Path to save the synthetic answers.')
def generate_synthetic_answers_cli(model_name, output_path):
    """Generates synthetic answers and saves them to a file."""
    client = config.get_client()
    result = load_data(config.XML_FILE)
    ground_truth = load_data(config.JSON_KEY_FILE)
    click.echo(f"Generating synthetic answers with model: {model_name}")
    generate_synthetic_answers(client, model_name, result, ground_truth, output_path)

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.5-pro', help='Model to use for generating reasoning.')
@click.option('--input', 'input_path', default=config.SYNTHETIC_ANSWERS_FILE, help='Path to the synthetic answers file.')
@click.option('--output', 'output_path', default=config.SYNTHETIC_ANSWERS_WITH_REASONING_FILE, help='Path to save the answers with reasoning.')
def generate_reasoning_cli(model_name, input_path, output_path):
    """Generates reasoning for synthetic answers and saves them to a file."""
    client = config.get_client()
    
    click.echo(f"Generating reasoning with model: {model_name}")
    generate_reasoning(client, model_name, input_path, output_path)

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.5-pro', help='Model to use for summarizing articles.')
@click.option('--output', 'output_path', default=config.SUMMARIZED_ARTICLES_FILE, help='Path to save the summarized articles.')
def summarize_articles_cli(model_name, output_path):
    """Summarizes articles and saves them to a file."""
    from rag.retrieval import build_faiss_index, embed_text, retrieve_articles
    from transformers import AutoModel, AutoTokenizer
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

    click.echo(f"Summarizing articles with model: {model_name}")
    summarize_articles(client, model_name, retrieved_articles, output_path)

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.5-pro', help='Model to use for generating synthetic cases.')
@click.option('--output', 'output_path', default=config.SYNTHETIC_CASES_FILE, help='Path to save the synthetic cases.')
def generate_synthetic_cases_cli(model_name, output_path):
    """Generates synthetic cases and saves them to a file."""
    from rag.retrieval import build_faiss_index, embed_text, retrieve_articles
    from transformers import AutoModel, AutoTokenizer
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

    click.echo(f"Generating synthetic cases with model: {model_name}")
    generate_synthetic_cases(client, model_name, retrieved_articles, output_path)

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.5-pro', help='Model to use for the RAG experiment.')
@click.option('--summarizer', 'summarizer_model', default='google/gemini-2.5-pro', help='Model to use for summarization.')
def rag(model_name, summarizer_model):
    """Runs the RAG experiment."""
    from rag.retrieval import build_faiss_index, embed_text, retrieve_articles
    from transformers import AutoModel, AutoTokenizer
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
@click.option('--model', 'model_name', default='google/gemini-2.5-pro', help='Model to use for the RAG with summary experiment.')
@click.option('--summarizer', 'summarizer_model', default='google/gemini-2.5-pro', help='Model to use for summarization.')
@click.pass_context
def rag_summary(ctx, model_name, summarizer_model):
    """Runs the RAG experiment with summarized articles."""
    if not os.path.exists(config.SUMMARIZED_ARTICLES_FILE):
        click.echo("Summarized articles file not found. Generating it now...")
        ctx.invoke(summarize_articles_cli, output_path=config.SUMMARIZED_ARTICLES_FILE)

    client = config.get_client()
    result = load_data(config.XML_FILE)
    with open(config.SUMMARIZED_ARTICLES_FILE, 'rb') as f:
        summarized_articles = pickle.load(f)
    with open('prompts/example.json', 'r') as f:
        example = json.load(f)[0]

    click.echo(f"Running RAG with summary experiment with model: {model_name}")
    rag2_answers = generate_answers(
        cases=result["cases"],
        client=client,
        model_name=model_name,
        prompt_fn=lambda case, idx: generate_prompt_rag_summary(case, summarized_articles[idx], example),
        max_retries=5,
        use_index=True,
    )
    prepare_submission_file(
        rag2_answers,
        f"{config.OUTPUT_DIR}/submission_rag_summary.json",
        result,
        client,
        summarizer_model
    )
    click.echo("RAG with summary experiment complete.")

@cli.command()
@click.option('--model', 'model_name', default='google/gemini-2.5-pro', help='Model to use for the RAG with synthetic cases experiment.')
@click.option('--summarizer', 'summarizer_model', default='google/gemini-2.5-pro', help='Model to use for summarization.')
@click.pass_context
def rag_synthetic_cases(ctx, model_name, summarizer_model):
    """Runs the RAG experiment with synthetic cases."""
    if not os.path.exists(config.SYNTHETIC_CASES_FILE):
        click.echo("Synthetic cases file not found. Generating it now...")
        ctx.invoke(generate_synthetic_cases_cli, output_path=config.SYNTHETIC_CASES_FILE)

    client = config.get_client()
    result = load_data(config.XML_FILE)
    df_synthetic = pd.read_csv(config.SYNTHETIC_CASES_FILE)
    all_synthetic_cases = df_synthetic.to_dict(orient="records")

    click.echo(f"Running RAG with synthetic cases experiment with model: {model_name}")
    rag3_answers = generate_answers(
        cases=result["cases"],
        client=client,
        model_name=model_name,
        prompt_fn=lambda case: generate_prompt_rag_synthetic_cases(case, all_synthetic_cases),
        max_retries=5
    )
    prepare_submission_file(
        rag3_answers,
        f"{config.OUTPUT_DIR}/submission_rag_synthetic_cases.json",
        result,
        client,
        summarizer_model
    )
    click.echo("RAG with synthetic cases experiment complete.")

@cli.command()
@click.pass_context
def all(ctx):
    """Runs all experiments with their default models."""
    ctx.invoke(baseline)
    ctx.invoke(few_shot_basic)
    ctx.invoke(few_shot_syn_ans)
    ctx.invoke(few_shot_syn_w_reasoning)
    ctx.invoke(rag)
    ctx.invoke(rag_summary)
    ctx.invoke(rag_synthetic_cases)
    click.echo("All experiments complete.")

if __name__ == '__main__':
    cli() 