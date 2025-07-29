import random

example1 = {
    "patient_narrative": "I had severe abdomen pain and was hospitalised for 15 days in ICU, diagnoised with CBD sludge. Doctor advised for ERCP. My question is if the sludge was there does not any medication help in flushing it out? Whether ERCP was the only cure?",
    "patient_question": "My question is if the sludge was there does not any medication help in flushing it out? Whether ERCP was the only cure?",
    "clinician_question": "Why was ERCP recommended over a medication-based treatment for CBD sludge?",
    "clinical_note": "1: During the ERCP a pancreatic stent was required to facilitate access to the biliary system (removed at the end of the procedure), and a common bile duct stent was placed to allow drainage of the biliary obstruction caused by stones and sludge.\n2: However, due to the patient's elevated INR, no sphincterotomy or stone removal was performed.\n3: Frank pus was noted to be draining from the common bile duct, and post-ERCP it was recommended that the patient remain on IV Zosyn for at least a week.\n4: The Vancomycin was discontinued.\n5: On hospital day 4 (post-procedure day 3) the patient returned to ERCP for re-evaluation of her biliary stent as her LFTs and bilirubin continued an upward trend.\n6: On ERCP the previous biliary stent was noted to be acutely obstructed by biliary sludge and stones.\n7: As the patient's INR was normalized to 1.2, a sphincterotomy was safely performed, with removal of several biliary stones in addition to the common bile duct stent.\n8: At the conclusion of the procedure, retrograde cholangiogram was negative for filling defects.",
    "relevant_sentences": "1: During the ERCP a pancreatic stent was required to facilitate access to the biliary system (removed at the end of the procedure), and a common bile duct stent was placed to allow drainage of the biliary obstruction caused by stones and sludge.\n5: On ERCP the previous biliary stent was noted to be acutely obstructed by biliary sludge and stones.\n6: As the patient's INR was normalized to 1.2, a sphincterotomy was safely performed, with removal of several biliary stones in addition to the common bile duct stent.\n7: At the conclusion of the procedure, retrograde cholangiogram was negative for filling defects.",
    "answer": "Medications can sometimes help in managing bile duct sludge, but in this case, ERCP was necessary due to the severity of the obstruction and its complications.\nThe initial ERCP revealed significant biliary obstruction caused by sludge and stones, requiring the placement of a stent to restore bile drainage |1|.\nHowever, even after this intervention, the liver function tests and bilirubin levels continued to rise, indicating that the obstruction was not fully resolved |5|.\nA follow-up ERCP confirmed that the stent itself had become acutely obstructed by sludge and stones, necessitating further intervention |6|.\nDuring this procedure, a sphincterotomy was performed, and several stones were physically removed, which medications alone could not have achieved |7|.\nThese findings confirm that ERCP was essential in addressing his condition and preventing further complications."
}

example2 = {
    "patient_narrative": "Took my 59 yo father to ER ultrasound discovered he had an aortic aneurysm. He had a salvage repair (tube graft). Long surgery / recovery for couple hours then removed packs. why did they do this surgery????? After this time he spent 1 month in hospital now sent home.",
    "patient_question": "Why did they do this surgery?",
    "clinician_question": "Why did they perform the emergency salvage repair on him?",
    "clinical_note": "1: He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm.\n2: He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest.\n3: Please see operative note for details which included cardiac arrest x2.\n4: Postoperatively he was taken to the intensive care unit for monitoring with an open chest.\n5: He remained intubated and sedated on pressors and inotropes.\n6: On 2025-1-22, he returned to the operating room where he underwent exploration and chest closure.\n7: On 1-25 he returned to the OR for abd closure JP/ drain placement/ feeding jejunostomy placed at that time for nutritional support.\n8: Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema. 9: Packed with dry gauze and covered w/DSD.",
    "relevant_sentences": "1: He was transferred to the hospital on 2025-1-20 for emergent repair of his ruptured thoracoabdominal aortic aneurysm.\n2: He was immediately taken to the operating room where he underwent an emergent salvage repair of ruptured thoracoabdominal aortic aneurysm with a 34-mm Dacron tube graft using deep hypothermic circulatory arrest.\n8: Thoracoabdominal wound healing well with exception of very small open area mid wound that is @1cm around and 1/2cm deep, no surrounding erythema. 9: Packed with dry gauze and covered w/DSD.",
    "answer": "His aortic aneurysm was caused by the rupture of a thoracoabdominal aortic aneurysm, which required emergent surgical intervention. |1|\n He underwent a complex salvage repair using a 34-mm Dacron tube graft and deep hypothermic circulatory arrest to address the rupture. |2|\n The extended recovery time and hospital stay were necessary due to the severity of the rupture and the complexity of the surgery, though his wound is now healing well with only a small open area noted. |8|"
}

def format_clinical_note(sentences):
    newline = '\n'
    return newline.join([f"{s['id']}: {s['text'].replace(newline, '')}" for s in sentences])

def generate_prompt_baseline(data):
    prompt = f"""
# Clinical Note Question Answering System
You are a medical expert tasked with providing clear, accurate answers to medical questions based on relevant sentences from the clinical notes.
Your response should be detailed, evidence-based, and reference specific points from the relevant sentences using the numbered citations.
You only allowed to use the relevant sentences to answer the question.

# To answer

Patient Narrative: {data['patient_narrative']}
Patient Question: {data['patient_questions'][0]['text']}
Clinician Question: {data['clinician_question']}
Clinical Note: {format_clinical_note(data['sentences'])}

Return your response in the below format strictly.

<plan>
look at the notes first explain all the disease/drugs/symptoms/etc in details.
make sure you understand the patient underlying conditions and the progression of the disease.
</plan>

<think>
your step-by-step thinking on how to formulate the answer think about the patient underlying conditions and the progression of the disease.
</think>

<answer>
# Guidelines for response:
1. Each sentence in your answer must reference specific evidence from the clinical note.
2. Include sentence numbers in format |#| after each evidence-based statement.
3. Maintain medical accuracy while using patient-friendly language.
4. Do not add information beyond what is in the clinical notes.
5. Return your response in the below format strictly.
your answer in the format of |#| new_line sentence. |reference| new_line sentence. |reference| new_line
Please do not use hyphen('-') in the citation. List all the citations.
</answer>
"""
    return prompt

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

    prompt = f"""
{''.join(example_prompts)}

# To answer

Patient Narrative: {data['patient_narrative']}
Patient Question: {data['patient_questions'][0]['text']}
Clinician Question: {data['clinician_question']}
Clinical Note: {format_clinical_note(data['sentences'])}

Return your response in the below format strictly.
<answer>
your answer based on the things you have seen in the Example Patient Narrative, Example Patient Question, Example Clinician Question, Example Clinical Note and the Example Answer.
Please do not use hyphen('-') in the citation. List all the citations. For multiple citation, please seperate it by comma e.g. |1, 2, 3|
</answer>
"""
    return prompt

def generate_prompt_rag(data, retrieved_articles, use_summaries=False):
    article_key = "summary" if use_summaries else "article"
    relevant_articles = '\n'.join([
        f"Title: {art['title']}\n{article_key.capitalize()}: {art[article_key]}"
        for art in retrieved_articles
    ])

    prompt = f"""
Articles:
{relevant_articles}

Clinical Note:
{format_clinical_note(data['sentences'])}

<answer>
# Guidelines for response:
1. Each sentence in your answer must reference specific evidence from the clinical note.
2. Include sentence numbers in format |#| after each evidence-based statement.
3. Maintain medical accuracy while using patient-friendly language.
4. Do not add information beyond what is in the clinical notes.
5. Return your response in the below format strictly.
your answer in the format of |#| new_line sentence. |reference| ...
</answer>
"""
    return prompt 