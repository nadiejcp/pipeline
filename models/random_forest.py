from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel

class RandomForestModel(BaseModel):
  def __init__(self, **kwargs):
    if 'random_state' not in kwargs:
      kwargs['random_state'] = 42
    self.model = RandomForestClassifier(**kwargs)

  def fit(self, X, y):
    self.model.fit(X, y)
    return self

  def predict(self, X):
    return self.model.predict(X)

  def predict_proba(self, X):
    return self.model.predict_proba(X)
