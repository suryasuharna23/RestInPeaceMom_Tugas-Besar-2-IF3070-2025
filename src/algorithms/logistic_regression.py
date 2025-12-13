import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, class_weight=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.class_weight = class_weight

        self.weights = None
        self.bias = None
        self.loss_history = []

    # =========================
    # Sigmoid aman secara numerik
    # =========================
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    # =========================
    # Loss dengan class_weight
    # =========================
    def _compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        if self.class_weight is None:
            weights = np.ones_like(y_true)
        else:
            weights = np.where(
                y_true == 1, 
                self.class_weight.get(1, 1.0), 
                self.class_weight.get(0, 1.0)
            )

        loss = -np.mean(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        return loss

    # =========================
    # Fit Model
    # =========================
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        # Hitung bobot sampel
        if self.class_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.where(
                y == 1, 
                self.class_weight.get(1, 1.0),
                self.class_weight.get(0, 1.0)
            )

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Weighted gradient
            diff = (y_pred - y) * sample_weight
            dw = (1 / n_samples) * np.dot(X.T, diff)
            db = (1 / n_samples) * np.sum(diff)

            # Update parameter
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    # =========================
    # Predict Probability
    # =========================
    def predict_proba(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred_prob = self._sigmoid(linear_model)
        return np.column_stack([1 - y_pred_prob, y_pred_prob])

    # =========================
    # Predict Class
    # =========================
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)

    # =========================
    # Accuracy Score
    # =========================
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    # =========================
    # Return Coefficients
    # =========================
    def get_coefficients(self):
        return {
            'weights': self.weights,
            'bias': self.bias
        }
