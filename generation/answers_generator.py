from tqdm.auto import tqdm
from utils.helpers import parse_prompt_response, check_exceed_max_number_of_citations, retry

@retry(max_retries=3, delay=1)
def generate_completion(client, model_name, prompt):
    return client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

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
        if use_index:
            prompt = prompt_fn(data, idx)
        else:
            prompt = prompt_fn(data)

        pred_text = None
        answer = None

        for attempt in range(max_retries):
            resp = generate_completion(client, model_name, prompt)
            pred_text = resp
            parsed = parse_prompt_response(pred_text)
            answer = parsed.get('answer')

            if answer and len(answer) >= 20 and not check_exceed_max_number_of_citations(answer):
                break

        answers.append({
            "case_id": case_id,
            "pred_text": pred_text,
            "answer": answer
        })

    return answers 