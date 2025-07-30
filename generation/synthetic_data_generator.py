import pandas as pd
from tqdm.auto import tqdm
from data_processing.loader import parse_referenced_answer
from evaluation.metrics import calculate_metrics
from utils.helpers import parse_prompt_response, retry
from generation.prompt_generator import generate_prompt_synthetic_answer, generate_prompt_reasoning
import pickle
import os
from generation.prompt_generator import generate_prompt_synthetic_case

def format_relevant_sentences(id, sentences):
    newline = '\n'
    return f"{id}: {sentences.strip().replace(newline, ' ')}"

@retry(max_retries=3, delay=1)
def generate_completion(client, model_name, prompt):
    return client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

def generate_answer_with_retry(client, model_name, data, ground_truth, max_retries=10, threshold=1):
    best_answer = None
    best_metrics = None
    best_f1 = 0
    attempts = 0

    for attempt in range(max_retries):
        attempts += 1
        prompt = generate_prompt_synthetic_answer(data)
        syn_answer = generate_completion(client, model_name, prompt)
        syn_answer = parse_prompt_response(syn_answer)["answer"]

        pred = parse_referenced_answer(syn_answer, case_id=data['id'],
                                    total_sentences=len(ground_truth['answers']))
        
        metrics = calculate_metrics(ground_truth, pred)

        if metrics['f1_score'] > best_f1:
            best_answer = syn_answer
            best_metrics = metrics
            best_f1 = metrics['f1_score']

        if best_f1 >= threshold:
            break

    return {
        'answer': best_answer,
        'metrics': best_metrics,
        'attempts': attempts
    }

def generate_synthetic_answers(client, model_name, result, ground_truth, output_path):
    answers = []
    ans_df = pd.DataFrame()

    for data in tqdm(result["cases"]):
        case_id = int(data['id'])
        gt = [g for g in ground_truth if int(g["case_id"]) == case_id][0]
        essential_sentences = [s["sentence_id"] for s in gt["answers"] if s["relevance"] == "essential"]
        
        data = {
            **data,
            "relevant_sentences": "\n".join([format_relevant_sentences(s["id"], s["text"])
                                           for s in data["sentences"] if s["id"] in essential_sentences])
        }

        result = generate_answer_with_retry(client, model_name, data, gt)
        syn_answer = result['answer']
        metrics = result['metrics']
        attempts = result['attempts']

        print(f"Generated in {attempts} attempts. F1 score: {metrics['f1_score']:.3f}")
        print(f"Predicted essential sentences: {metrics['prediction_essential']}")

        ans_df = pd.concat([ans_df, pd.DataFrame({
            "case_id": [case_id],
            "gt": [gt],
            "data": [data],
            "syn_answer": [syn_answer],
            "metrics": [metrics],
            "attempts": [attempts]
        })], ignore_index=True)

    ans_df.to_pickle(output_path)
    print(f"Synthetic answers saved to {output_path}")

def generate_reasoning(client, model_name, input_path, output_path):
    df = pd.read_pickle(input_path)
    all_cases = df.to_dict(orient="records")
    reasoning_list = []

    for data in tqdm(all_cases):
        prompt = generate_prompt_reasoning(data['data'])
        reasoning = generate_completion(client, model_name, prompt)
        reasoning_list.append(reasoning)
        
    df['syn_think'] = reasoning_list
    df.to_pickle(output_path)
    print(f"Reasoning saved to {output_path}")

def summarize_articles(client, model_name, retrieved_articles, output_path):
    retrieved_articles_per_query_sum = []
    for articles in tqdm(retrieved_articles, desc="Summarizing queries"):
        summarized_articles = []
        for article_data in tqdm(articles, desc="Summarizing articles", leave=False):
            title = article_data['title']
            content = article_data['article']
            prompt = f"Summarize the following medical article in 3–5 sentences:\n\n{content}"
            response = generate_completion(client, model_name, prompt)
            summarized_articles.append({
                'title': title,
                'summary': response.strip()
            })
        retrieved_articles_per_query_sum.append(summarized_articles)
    
    with open(output_path, 'wb') as f:
        pickle.dump(retrieved_articles_per_query_sum, f)
    print(f"Summarized articles saved to {output_path}")

def generate_synthetic_cases(client, model_name, retrieved_articles, output_path):
    checkpoint_file = output_path

    def load_checkpoint(file_path):
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty:
                last_index = df['global_index'].max() + 1
                return df.to_dict('records'), last_index
        return [], 0

    def save_checkpoint(data, file_path):
        if data:
            pd.DataFrame(data).to_csv(file_path, index=False)

    synthetic_data, start_index = load_checkpoint(checkpoint_file)

    total_articles = len(retrieved_articles) * 5

    with tqdm(total=total_articles, initial=start_index, desc="Generating QA") as pbar:
        for case_index, articles in enumerate(retrieved_articles):
            for article_index, article_dict in enumerate(articles):
                global_index = case_index * 5 + article_index
                if global_index < start_index:
                    continue

                title = article_dict['title']
                content = article_dict['article']

                prompt = generate_prompt_synthetic_case(title, content)

                try:
                    response = generate_completion(client, model_name, prompt)
                    synthetic_data.append({
                        'global_index': global_index,
                        'case': f"case_{case_index+1}",
                        'article_index': article_index + 1,
                        'title': title,
                        'article': content,
                        'synthetic_qa': response
                    })
                    save_checkpoint(synthetic_data, checkpoint_file)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error at case {case_index}, article {article_index}: {e}")
                    continue
    print(f"Synthetic cases saved to {output_path}") 