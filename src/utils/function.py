"""
Custom evaluation metrics implementation from scratch
For fraud detection model evaluation
"""

import numpy as np


def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy score
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    float : Accuracy score
    """
    return np.mean(y_true == y_pred)


def calculate_precision(y_true, y_pred):
    """
    Calculate precision score
    Precision = TP / (TP + FP)
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    float : Precision score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def calculate_recall(y_true, y_pred):
    """
    Calculate recall score (sensitivity)
    Recall = TP / (TP + FN)
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    float : Recall score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def calculate_f1_score(precision, recall):
    """
    Calculate F1-score
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Parameters:
    -----------
    precision : float
        Precision score
    recall : float
        Recall score
        
    Returns:
    --------
    float : F1-score
    """
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix
    
    Returns 2x2 matrix:
    [[TN, FP],
     [FN, TP]]
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    numpy.ndarray : 2x2 confusion matrix
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def calculate_roc_auc(y_true, y_pred_proba):
    """
    Calculate ROC-AUC score and ROC curve coordinates
    Uses trapezoidal rule for AUC calculation
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
        
    Returns:
    --------
    tuple : (auc_score, fpr_array, tpr_array)
        - auc_score: Area under ROC curve
        - fpr_array: False positive rates
        - tpr_array: True positive rates
    """
    y_true_arr = np.array(y_true)
    thresholds = np.unique(y_pred_proba)
    thresholds = np.sort(thresholds)[::-1]

    tpr = []
    fpr = []
    
    for threshold in thresholds:
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_true_arr == 1) & (y_pred_temp == 1))
        fp = np.sum((y_true_arr == 0) & (y_pred_temp == 1))
        tn = np.sum((y_true_arr == 0) & (y_pred_temp == 0))
        fn = np.sum((y_true_arr == 1) & (y_pred_temp == 0))

        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr.append(tpr_val)
        fpr.append(fpr_val)

    tpr = np.array(tpr)
    fpr = np.array(fpr)
    
    # Sort by FPR
    sort_idx = np.argsort(fpr)
    fpr = fpr[sort_idx]
    tpr = tpr[sort_idx]

    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    return auc, fpr, tpr


def find_optimal_threshold(y_true, y_pred_proba, metric='f1', thresholds=None):
    """
    Find optimal threshold for classification based on specified metric
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    metric : str, default='f1'
        Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
    thresholds : array-like, optional
        Thresholds to test. If None, uses np.linspace(0.01, 0.99, 99)
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'threshold': optimal threshold value
        - 'score': score at optimal threshold
        - 'precision': precision at optimal threshold
        - 'recall': recall at optimal threshold
        - 'f1': F1-score at optimal threshold
        - 'accuracy': accuracy at optimal threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    best_score = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
        
        tp = np.sum((y_pred_temp == 1) & (y_true == 1))
        fp = np.sum((y_pred_temp == 1) & (y_true == 0))
        fn = np.sum((y_pred_temp == 0) & (y_true == 1))
        tn = np.sum((y_pred_temp == 0) & (y_true == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Select score based on metric
        if metric == 'f1':
            score = f1
        elif metric == 'precision':
            score = precision
        elif metric == 'recall':
            score = recall
        elif metric == 'accuracy':
            score = accuracy
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'f1', 'precision', 'recall', or 'accuracy'")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'score': score,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
    
    return best_metrics


def print_classification_report(y_true, y_pred, y_pred_proba=None):
    """
    Print comprehensive classification report
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities (for ROC-AUC)
    """
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1 = calculate_f1_score(precision, recall)
    cm = calculate_confusion_matrix(y_true, y_pred)
    
    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    if y_pred_proba is not None:
        auc, _, _ = calculate_roc_auc(y_true, y_pred_proba)
        print(f"ROC-AUC:   {auc:.4f}")
    
    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives:  {cm[1, 1]}")
    print("=" * 50)
