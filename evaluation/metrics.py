def calculate_metrics(ground_truth, prediction):
    # Extract sentence IDs marked as essential in ground truth
    gt_essential = set()
    for answer in ground_truth['answers']:
        if answer['relevance'] == 'essential':
            gt_essential.add(answer['sentence_id'])

    # Extract sentence IDs marked as essential in prediction
    pred_essential = set()
    for answer in prediction['answers']:
        if answer['relevance'] == 'essential':
            pred_essential.add(answer['sentence_id'])

    # Calculate metrics
    true_positives = len(gt_essential.intersection(pred_essential))
    false_positives = len(pred_essential - gt_essential)
    false_negatives = len(gt_essential - pred_essential)

    # Avoid division by zero
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'case_id': ground_truth['case_id'],
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'ground_truth_essential': sorted(list(gt_essential)),
        'prediction_essential': sorted(list(pred_essential))
    } 