import random
import json

def load_prompt(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def format_clinical_note(sentences):
    newline = '\n'
    return newline.join([f"{s['id']}: {s['text'].replace(newline, '')}" for s in sentences])

def generate_prompt_baseline(data):
    prompt_template = load_prompt('prompts/baseline.txt')
    return prompt_template.format(
        patient_narrative=data['patient_narrative'],
        patient_question=data['patient_questions'][0]['text'],
        clinician_question=data['clinician_question'],
        clinical_note=format_clinical_note(data['sentences'])
    )

def generate_prompt_fewshot(data, examples, add_reasoning=False):
    example_prompts = []
    for ex in examples:
        reasoning = f"Example Reasoning/Thinking: {ex['syn_think']}\n" if add_reasoning else ""
        example_prompts.append(f"""
# Example

Example Patient Narrative: {ex['patient_narrative']}
Example Patient Question: {ex['patient_questions'][0]['text']}
Example Clinician Question: {ex['clinician_question']}
Example Clinical Note: {format_clinical_note(ex['sentences'])}
{reasoning}
Example Answer: {ex["syn_answer"]}
""")
    
    prompt_template = load_prompt('prompts/fewshot.txt')
    return prompt_template.format(
        examples=''.join(example_prompts),
        patient_narrative=data['patient_narrative'],
        patient_question=data['patient_questions'][0]['text'],
        clinician_question=data['clinician_question'],
        clinical_note=format_clinical_note(data['sentences'])
    )

def generate_prompt_rag(data, retrieved_articles, use_summaries=False):
    article_key = "summary" if use_summaries else "article"
    relevant_articles = '\n'.join([
        f"Title: {art['title']}\n{article_key.capitalize()}: {art[article_key]}"
        for art in retrieved_articles
    ])
    
    prompt_template = load_prompt('prompts/rag.txt')
    return prompt_template.format(
        articles=relevant_articles,
        clinical_note=format_clinical_note(data['sentences'])
    ) 