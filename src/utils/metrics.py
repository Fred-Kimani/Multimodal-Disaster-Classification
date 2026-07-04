from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix

def calculate_metrics(y_true, y_pred) -> dict:
    """
    Computes classification metrics: Accuracy, Precision, Recall, F1, Macro F1,
    Balanced Accuracy, and Confusion Matrix.
    
    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.
        
    Returns:
        dict: Containing calculated metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, and f1 for macro averaging
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Calculate precision, recall, and f1 for weighted averaging
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist()
    }
