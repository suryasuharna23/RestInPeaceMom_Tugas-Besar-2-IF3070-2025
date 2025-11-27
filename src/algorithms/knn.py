import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class KNN:
    """
    K-Nearest Neighbors Classifier
    
    Parameters:
    -----------
    k : int, default=5
        Number of neighbors to use
    metric : str, default='euclidean'
        Distance metric to use ('euclidean' or 'manhattan')
    """
    
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Fit the KNN model with training data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _calculate_distance(self, x1, x2):
        """
        Calculate distance between two points
        
        Parameters:
        -----------
        x1 : array-like
            First point
        x2 : array-like
            Second point
            
        Returns:
        --------
        float
            Distance between x1 and x2
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _get_neighbors(self, x):
        """
        Get k nearest neighbors for a single point
        
        Parameters:
        -----------
        x : array-like
            Query point
            
        Returns:
        --------
        array-like
            Labels of k nearest neighbors
        """
        # Calculate distances to all training points
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._calculate_distance(x, x_train)
            distances.append((dist, self.y_train[i]))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:self.k]]
        
        return k_nearest
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            neighbors = self._get_neighbors(x)
            # Majority voting
            most_common = Counter(neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        array-like, shape (n_samples, 2)
            Predicted class probabilities [prob_class_0, prob_class_1]
        """
        X = np.array(X)
        probabilities = []
        
        for x in X:
            neighbors = self._get_neighbors(x)
            # Calculate probability as proportion of positive class
            prob_positive = sum(neighbors) / len(neighbors)
            probabilities.append([1 - prob_positive, prob_positive])
        
        return np.array(probabilities)
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True labels
            
        Returns:
        --------
        float
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def evaluate(self, X, y):
        """
        Evaluate model with multiple metrics
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True labels
            
        Returns:
        --------
        dict
            Dictionary containing various evaluation metrics
        """
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def get_params(self):
        """
        Get model parameters
        
        Returns:
        --------
        dict
            Dictionary of model parameters
        """
        return {
            'k': self.k,
            'metric': self.metric
        }
    
    def set_params(self, **params):
        """
        Set model parameters
        
        Parameters:
        -----------
        **params : dict
            Model parameters to set
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


def tune_knn_hyperparameters(X_train, y_train, X_val, y_val, 
                             k_values=[3, 5, 7, 9, 11, 15, 21],
                             metrics=['euclidean', 'manhattan'],
                             scoring='f1_score'):
    """
    Tune KNN hyperparameters using validation set
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    k_values : list, default=[3, 5, 7, 9, 11, 15, 21]
        List of k values to try
    metrics : list, default=['euclidean', 'manhattan']
        List of distance metrics to try
    scoring : str, default='f1_score'
        Metric to optimize ('accuracy', 'precision', 'recall', 'f1_score', 'roc_auc')
        
    Returns:
    --------
    tuple
        (best_model, best_params, results_df)
    """
    import pandas as pd
    
    results = []
    best_score = -1
    best_model = None
    best_params = None
    
    print("=" * 60)
    print("KNN HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Scoring metric: {scoring}")
    print(f"Testing {len(k_values)} k values × {len(metrics)} metrics = {len(k_values) * len(metrics)} combinations\n")
    
    total_combinations = len(k_values) * len(metrics)
    current = 0
    
    for k in k_values:
        for metric in metrics:
            current += 1
            print(f"[{current}/{total_combinations}] Testing k={k}, metric={metric}...", end=" ")
            
            # Train model
            model = KNN(k=k, metric=metric)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            metrics_dict = model.evaluate(X_val, y_val)
            score = metrics_dict[scoring]
            
            print(f"{scoring}={score:.4f}")
            
            # Store results
            results.append({
                'k': k,
                'metric': metric,
                'accuracy': metrics_dict['accuracy'],
                'precision': metrics_dict['precision'],
                'recall': metrics_dict['recall'],
                'f1_score': metrics_dict['f1_score'],
                'roc_auc': metrics_dict['roc_auc']
            })
            
            # Update best model
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'k': k, 'metric': metric}
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(scoring, ascending=False)
    
    print("\n" + "=" * 60)
    print("BEST PARAMETERS FOUND:")
    print("=" * 60)
    print(f"k = {best_params['k']}")
    print(f"metric = {best_params['metric']}")
    print(f"Best {scoring}: {best_score:.4f}")
    print("=" * 60)
    
    return best_model, best_params, results_df


def evaluate_knn_model(model, X_train, y_train, X_val, y_val, model_name="KNN"):
    """
    Comprehensive evaluation of KNN model
    
    Parameters:
    -----------
    model : KNN
        Trained KNN model
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    model_name : str, default="KNN"
        Name of the model for display
        
    Returns:
    --------
    dict
        Dictionary containing training and validation metrics
    """
    print("\n" + "=" * 60)
    print(f"{model_name} MODEL EVALUATION")
    print("=" * 60)
    
    # Evaluate on training set
    print("\nTraining Set Performance:")
    train_metrics = model.evaluate(X_train, y_train)
    for metric, value in train_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Evaluate on validation set
    print("\nValidation Set Performance:")
    val_metrics = model.evaluate(X_val, y_val)
    for metric, value in val_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print("=" * 60)
    
    return {
        'train': train_metrics,
        'validation': val_metrics
    }
