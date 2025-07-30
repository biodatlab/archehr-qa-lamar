import re
import time
from functools import wraps

def retry(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator

def parse_prompt_response(response_text):
    answer_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
    answer_match = re.search(answer_pattern, response_text)
    answer = answer_match.group(1).strip() if answer_match else ""
    return {"answer": answer}

def add_newlines_after_references(text):
    pattern = r'(\|\d+(?:,\s*\d+)*\|)'
    result = re.sub(pattern, r'\1\n', text)
    return result

def check_exceed_max_number_of_citations(answer_text):
    reference_patterns = re.findall(r'\|([0-9,\s]+)\|', answer_text)
    for ref_pattern in reference_patterns:
        if len(ref_pattern.split(',')) > 20:
            return True
    return False 