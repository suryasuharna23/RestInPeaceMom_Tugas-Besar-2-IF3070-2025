import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization=None, lambda_reg=0.01, 
                 tol=1e-4, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _sigmoid(self, z):
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)), 
                        np.exp(z) / (1 + np.exp(z)))
    
    def _compute_loss(self, y_true, y_pred, linear_model):
        m = len(y_true)
        
        loss = np.mean(np.maximum(linear_model, 0) - linear_model * y_true + np.log(1 + np.exp(-np.abs(linear_model))))
        
        if self.regularization == 'l2':
            loss += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            loss += (self.lambda_reg / m) * np.sum(np.abs(self.weights))
        
        return loss
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).ravel()
        
        n_samples, n_features = X.shape
        
        limit = 1 / np.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, (n_features,))
        self.bias = 0
        self.loss_history = []
        
        prev_loss = float('inf')
        
        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            loss = self._compute_loss(y, y_pred, linear_model)
            self.loss_history.append(loss)
            
            if self.verbose and (i % 100 == 0):
                print(f"Iteration {i}: Loss = {loss:.6f}")
            
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i}. Loss: {loss:.6f}")
                break
            prev_loss = loss
        
        return self
    
    def predict_proba(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred_prob = self._sigmoid(linear_model)
        return np.column_stack([1 - y_pred_prob, y_pred_prob])
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        y = np.array(y).ravel()
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_coefficients(self):
        return {
            'weights': self.weights,
            'bias': self.bias
        } 