import torch
from torch.nn import CrossEntropyLoss


def calculate_loss(logits, labels, intervenable, intervention_type):
    """
    Calculate loss matching the notebook's approach.
    """
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, intervenable.model_config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    # Add boundary loss for boundless interventions (commented out in notebook)
    # if intervention_type == "boundless":
    #     for k, v in intervenable.interventions.items():
    #         boundary_loss = 1.0 * v.intervention_boundaries.sum()
    #     loss += boundary_loss

    return loss


def compute_metrics(eval_preds, eval_labels):
    """
    Compute metrics matching the notebook's approach.
    Compares the last token predictions with the true labels.
    """
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -1]
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
        correct_labels = actual_test_labels == pred_test_labels
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
    accuracy = round(correct_count / total_count, 2)
    return {"accuracy": accuracy}