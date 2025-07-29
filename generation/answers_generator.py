from tqdm.auto import tqdm
import json
from time import sleep
from utils.helpers import parse_prompt_response, check_exceed_max_number_of_citations
import os

def generate_answers(
    cases,
    client,
    model_name,
    prompt_fn,
    max_retries=3,
    use_index=False
):
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
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
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

    return answers 