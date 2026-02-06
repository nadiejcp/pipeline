from sklearn.neural_network import MLPClassifier
from .base import BaseModel
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class TorchMLP(nn.Module):
  def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.0):
    super(TorchMLP, self).__init__()
    layers = []
    in_dim = input_dim
    for h_dim in hidden_layers:
      layers.append(nn.Linear(in_dim, h_dim))
      layers.append(nn.ReLU())
      if dropout > 0:
        layers.append(nn.Dropout(dropout))
      in_dim = h_dim
    layers.append(nn.Linear(in_dim, output_dim))
    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)

class MLPModel(BaseModel):
  def __init__(self, hidden_layers=None, dropout=None, device=None, **kwargs):
    """
    Initialize MLP Model.
    Args:
        hidden_layers (list): List of neuron counts for hidden layers.
        dropout (float): Dropout rate.
        device (torch.device): Device to run on (cpu or cuda).
        **kwargs: Additional arguments for MLPClassifier or training loop.
    """
    self.hidden_layers = hidden_layers
    self.dropout = dropout
    self.device = device
    self.kwargs = kwargs

    # Clean kwargs for sklearn
    sklearn_kwargs = kwargs.copy()
    sklearn_kwargs.pop('device', None) # Remove device explicitly
    if hidden_layers:
      sklearn_kwargs['hidden_layer_sizes'] = tuple(hidden_layers)
    if 'random_state' not in sklearn_kwargs:
      sklearn_kwargs['random_state'] = 42
    if 'max_iter' not in sklearn_kwargs:
      sklearn_kwargs['max_iter'] = 500

    self.sklearn_model = MLPClassifier(**sklearn_kwargs)
    self.torch_model = None
    self.classes_ = None

  def fit(self, X, y):
    if self.device:
      self._fit_torch(X, y)
    else:
      self.sklearn_model.fit(X, y)
    return self

  def _fit_torch(self, X, y):
    print(f"Training MLP on {self.device}...")
    
    # Store classes
    self.classes_ = np.unique(y)
    output_dim = len(self.classes_)
    
    # Prepare data
    y_map = {label: i for i, label in enumerate(self.classes_)}
    y_enc = np.vectorize(y_map.get)(y)
    
    X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
    y_t = torch.tensor(y_enc, dtype=torch.long).to(self.device)
    
    # Training params
    batch_size = self.kwargs.get('batch_size', 32)
    if batch_size == 'auto': batch_size = 32
    lr = self.kwargs.get('learning_rate_init', 0.001)
    max_iter = self.kwargs.get('max_iter', 200)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = X.shape[1]
    hidden_layout = self.hidden_layers if self.hidden_layers else (100,)
    
    self.torch_model = TorchMLP(input_dim, hidden_layout, output_dim, self.dropout if self.dropout else 0.0).to(self.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.torch_model.parameters(), lr=lr)

    self.torch_model.train()
    for epoch in tqdm(range(max_iter), desc=f"Training MLP"):
      for xb, yb in loader:
        optimizer.zero_grad()
        out = self.torch_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

  def predict(self, X):
    if self.torch_model:
      self.torch_model.eval()
      with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        out = self.torch_model(X_t)
        _, preds = torch.max(out, 1)
        return self.classes_[preds.cpu().numpy()]
    else:
      return self.sklearn_model.predict(X)

  def predict_proba(self, X):
    if self.torch_model:
      self.torch_model.eval()
      with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        out = self.torch_model(X_t)
        probs = torch.softmax(out, dim=1)
        return probs.cpu().numpy()
    else:
      return self.sklearn_model.predict_proba(X)
