import re
import json
from tqdm.auto import tqdm
import os
from generation.prompt_generator import load_prompt

def get_ref_ids(answer_text):
    references = set()
    reference_patterns = re.findall(r'\|([0-9,\s]+)\|', answer_text)
    for ref_pattern in reference_patterns:
        for ref in ref_pattern.split(','):
            try:
                sentence_id = ref.strip()
                if sentence_id.isdigit():
                    references.add(sentence_id)
            except ValueError:
                pass
    return references

def format_sentences_with_ids(sentences, sentence_ids):
    result = ""
    sentence_ids = sorted([int(id_) for id_ in sentence_ids])
    for sentence_id in sentence_ids:
        for sentence in sentences:
            if sentence['id'] == str(sentence_id):
                result += f"{sentence_id}: {sentence['text']}\n"
                break
    return result.strip()

def craft_summarize_prompt(text, sentences):
    sentence_ids = get_ref_ids(text)
    prompt_template = load_prompt('prompts/summarize.txt')
    return prompt_template.format(
        referenced_sentences=format_sentences_with_ids(sentences, sentence_ids),
        text=text
    )

def summarize_text(text, sentences, client, model_name):
    prior_ids = get_ref_ids(text)
    prompt = craft_summarize_prompt(text, sentences)
    best_response = None
    best_score = 0
    retries = 0
    max_retries = 5
    while retries < max_retries:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
        new_ids = get_ref_ids(response)
        matching_ids = len(prior_ids.intersection(new_ids))
        total_ids = len(prior_ids.union(new_ids))
        score = matching_ids / total_ids if total_ids > 0 else 0
        if score == 1.0:
            return response
        if score > best_score:
            best_score = score
            best_response = response
        retries += 1
    return best_response if best_response else response

def clean_answer(answer_text):
    matches = re.findall(r'.*?\|\s*\d+(?:\s*,\s*\d+)*\s*\|', answer_text)
    cleaned_lines = []
    for i, line in enumerate(matches):
        line = re.sub(r'\|\s*([\d,\s]+?)\s*\|', lambda m: '|' + m.group(1).replace(' ', '') + '|', line)
        line = re.sub(r'\s+(\|\d+(?:,\d+)*\|)', r'\1', line)
        line = re.sub(r'^[\s.,?!:;]+', '', line)
        if i < len(matches) - 1:
            line += '\n'
        cleaned_lines.append(line)
    return ''.join(cleaned_lines)

def prepare_submission_file(answers, submission_output_path, result, client, model_name):
    submission = answers
    for case in submission:
        for result_case in result['cases']:
            if case['case_id'] == result_case['id']:
                case['sentences'] = result_case['sentences']
    for case in tqdm(submission):
        summary = summarize_text(case['answer'], case['sentences'], client, model_name)
        case['raw_answer'] = case['answer']
        case['summarized_answer'] = summary
        case['answer'] = clean_answer(summary)
    for item in submission:
        item['case_id'] = str(item['case_id'])
        item['answer'] = clean_answer(item['answer'])
        
    output_dir = os.path.dirname(submission_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(submission_output_path, 'w') as f:
        json.dump(submission, f, indent=4) 