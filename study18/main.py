import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def evaluate_with_metrics(model, loader):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in loader:
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
            )

            logits = outputs.logits
            predictions = logits.argmax(dim=1)

            total_loss += outputs.loss.item() * batch_labels.size(0)
            total_samples += batch_labels.size(0)

            all_labels.extend(batch_labels.tolist())
            all_predictions.extend(predictions.tolist())

    average_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
    }