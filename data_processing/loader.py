from lxml import etree
from io import StringIO
import re
import json

def parse_clinical_annotations(xml_string):
    """
    Parse clinical annotations XML data into a dictionary format
    """
    # Create a parser that handles CDATA sections
    parser = etree.XMLParser(strip_cdata=False, recover=True)

    # Parse the XML string
    tree = etree.parse(StringIO(xml_string), parser)
    root = tree.getroot()

    # Create the result dictionary
    result = {
        "cases": []
    }

    # Process each case
    for case_elem in root.findall('.//case'):
        case = {
            "id": case_elem.get('id'),
            "patient_narrative": case_elem.findtext('patient_narrative', '').strip(),
            "patient_questions": [],
            "clinician_question": case_elem.findtext('clinician_question', '').strip(),
            "note_excerpt": case_elem.findtext('note_excerpt', '').strip(),
            "sentences": []
        }

        # Process patient questions
        for question_elem in case_elem.findall('.//patient_question/phrase'):
            case["patient_questions"].append({
                "id": question_elem.get('id'),
                "start_char_index": question_elem.get('start_char_index'),
                "text": question_elem.text.strip() if question_elem.text else ""
            })

        # Process sentences
        for sentence_elem in case_elem.findall('.//note_excerpt_sentences/sentence'):
            case["sentences"].append({
                "id": sentence_elem.get('id'),
                "paragraph_id": sentence_elem.get('paragraph_id'),
                "start_char_index": sentence_elem.get('start_char_index'),
                "text": sentence_elem.text.strip() if sentence_elem.text else ""
            })

        # Add the case to the result
        result["cases"].append(case)

    return result

def parse_referenced_answer(answer_text, case_id, total_sentences=9):
    """
    Parse an answer text with references and convert to structured JSON format.

    Args:
        answer_text (str): The answer text with references in format |#|
        case_id (str): The case ID for the output
        total_sentences (int): Total number of sentences in the clinical note

    Returns:
        dict: Structured JSON with sentence relevance information
    """
    # Extract all references from the answer text
    references = set()
    # Look for patterns like |1| or |5, 7|
    reference_patterns = re.findall(r'\|([0-9,\s]+)\|', answer_text)

    for ref_pattern in reference_patterns:
        # Handle multiple references like "5, 7"
        for ref in ref_pattern.split(','):
            try:
                sentence_id = ref.strip()
                if sentence_id.isdigit():
                    references.add(sentence_id)
            except ValueError:
                pass

    # Create the answers structure
    answers = []
    for i in range(total_sentences):
        sentence_id = str(i)
        relevance = "essential" if sentence_id in references else "not-relevant"
        answers.append({
            "sentence_id": sentence_id,
            "relevance": relevance
        })

    # Create the final structure
    result = {
        "case_id": case_id,
        "answers": answers
    }

    return result

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.xml'):
            return parse_clinical_annotations(f.read())
        elif file_path.endswith('.json'):
            return json.load(f)
    return None 