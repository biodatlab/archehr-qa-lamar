from tqdm.auto import tqdm
import json
from time import sleep
from utils.helpers import parse_prompt_response, check_exceed_max_number_of_citations
import os

def generate_answers(
    cases,
    model,
    prompt_fn,
    answers_json_path,
    max_retries=3,
    use_index=False
):
    """
    Iterate through cases, generate prompts, call the model, parse and save answers.

    Args:
        cases (List[dict]): List of case dicts.
        model: An LLM interface with `.generate_content(prompt).text`.
        prompt_fn (callable): Function to build a prompt; signature depends on `use_index`.
        answers_json_path (str): Path for full answers JSON (with texts).
        max_retries (int): Max number of generation attempts per case.
        use_index (bool): If True, calls prompt_fn(data, idx), else prompt_fn(data).
    Returns:
        List[dict]: The list of answer dicts with case_id, pred_text, and answer.
    """
    answers = []

    for idx, data in enumerate(tqdm(cases, desc="Generating answers")):
        case_id = data.get('id')
        # Build the prompt
        if use_index:
            prompt = prompt_fn(data, idx)
        else:
            prompt = prompt_fn(data)

        pred_text = None
        answer = None

        for attempt in range(max_retries):
            # Generate from the model
            resp = model.generate_content(prompt).text
            pred_text = resp
            parsed = parse_prompt_response(pred_text)
            answer = parsed.get('answer')

            # Check quality
            if answer and len(answer) >= 20 and not check_exceed_max_number_of_citations(answer):
                break
            sleep(4)

        answers.append({
            "case_id": case_id,
            "pred_text": pred_text,
            "answer": answer
        })

    # Save outputs
    output_dir = os.path.dirname(answers_json_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(answers_json_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=2)

    return answers 