from sklearn.neural_network import MLPClassifier
from .base import BaseModel
import warnings

class MLPModel(BaseModel):
  def __init__(self, hidden_layers=None, dropout=None, **kwargs):
    """
    Initialize MLP Model.
    Args:
        hidden_layers (list): List of neuron counts for hidden layers.
        dropout (float): Dropout rate (Ignored for sklearn MLPClassifier, but kept for interface compatibility).
        **kwargs: Additional arguments for MLPClassifier.
    """
    if hidden_layers:
      kwargs['hidden_layer_sizes'] = tuple(hidden_layers)
    if dropout is not None:
      warnings.warn(f"Dropout parameter ({dropout}) is currently ignored as sklearn.neural_network.MLPClassifier does not support it directly.")
    if 'random_state' not in kwargs:
      kwargs['random_state'] = 42
    if 'max_iter' not in kwargs:
      kwargs['max_iter'] = 1500

    self.model = MLPClassifier(**kwargs)

  def fit(self, X, y):
    self.model.fit(X, y)
    return self

  def predict(self, X):
    return self.model.predict(X)

  def predict_proba(self, X):
    return self.model.predict_proba(X)
