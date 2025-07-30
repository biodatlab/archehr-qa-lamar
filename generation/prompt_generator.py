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

def generate_prompt_synthetic_answer(data):
    prompt_template = load_prompt('prompts/synthetic_answer_generation.txt')
    return prompt_template.format(
        patient_narrative=data['patient_narrative'],
        patient_question=data['patient_questions'][0]['text'],
        clinician_question=data['clinician_question'],
        clinical_note=format_clinical_note(data['sentences']),
        relevant_sentences=data['relevant_sentences']
    )

def generate_prompt_reasoning(data):
    prompt_template = load_prompt('prompts/reasoning_generation.txt')
    return prompt_template.format(
        patient_narrative=data['patient_narrative'],
        patient_question=data['patient_questions'][0]['text'],
        clinician_question=data['clinician_question'],
        clinical_note=format_clinical_note(data['sentences']),
        answer=data['syn_answer']
    ) 

def generate_prompt_rag_summary(data, retrieved_articles, example):
    prompt_template = load_prompt('prompts/rag_summary.txt')
    return prompt_template.format(
        patient_narrative=data['patient_narrative'],
        patient_question=data['patient_questions'][0]['text'],
        clinician_question=data['clinician_question'],
        clinical_note=format_clinical_note(data['sentences']),
        relevant_sentences=example['relevant_sentences'],
        answer=example['answer'],
        relevant_article="\n".join([
            f"Title: {article['title']}\nArticle: {article['summary']}"
            for article in retrieved_articles
        ])
    )

def generate_prompt_synthetic_case(title, content):
    prompt_template = load_prompt('prompts/synthetic_case_generation.txt')
    return prompt_template.format(title=title, content=content)

def generate_prompt_rag_synthetic_cases(current_case, all_cases, test_case=None):
    from data_processing.loader import parse_synthetic_qa

    patient_narrative = test_case['patient_narrative']
    patient_question = test_case['patient_questions'][0]['text']
    clinical_question = test_case['clinician_question']
    clinical_note = '\n'.join([f"{s['id']}: {s['text']}" for s in test_case['sentences']])

    few_shot_examples = [case for case in all_cases][:5]

    example_texts = []
    for ex in few_shot_examples:
        parsed = parse_synthetic_qa(ex['synthetic_qa'])
        example_texts.append(f"""
Example Patient Question:
{parsed.get('Patient Question', '')}

Example Clinician Question:
{parsed.get('Clinician Question', '')}

Example Clinical Note:
{parsed.get('Clinical Note', '')}

Example Answer:
{parsed.get('Answer', '')}
""")

    examples_text = "\n# Example\n".join(example_texts)

    prompt_template = load_prompt('prompts/rag_synthetic.txt')
    return prompt_template.format(
        examples=examples_text,
        patient_narrative=patient_narrative,
        patient_question=patient_question,
        clinician_question=clinical_question,
        clinical_note=clinical_note
    ) 