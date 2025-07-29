import re

def parse_prompt_response(response_text):
    """
    Parse the response text from the prompt and extract the final answer.

    Args:
        response_text (str): The response text from the prompt

    Returns:
        dict: Structured JSON with the final answer
    """
    answer_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
    answer_match = re.search(answer_pattern, response_text)
    answer = answer_match.group(1).strip() if answer_match else ""
    return {"answer": answer}

def add_newlines_after_references(text):
    """
    Add newlines after each reference pattern |#| in the text
    """
    pattern = r'(\|\d+(?:,\s*\d+)*\|)'
    result = re.sub(pattern, r'\1\n', text)
    return result

def check_exceed_max_number_of_citations(answer_text):
    reference_patterns = re.findall(r'\|([0-9,\s]+)\|', answer_text)
    for ref_pattern in reference_patterns:
        if len(ref_pattern.split(',')) > 20:
            return True
    return False 