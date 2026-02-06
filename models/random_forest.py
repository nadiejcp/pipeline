from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel

class RandomForestModel(BaseModel):
  def __init__(self, **kwargs):
    if 'random_state' not in kwargs:
      kwargs['random_state'] = 42
    if 'max_iter' not in kwargs:
      kwargs['max_iter'] = 1000
    if 'n_estimators' not in kwargs:
      kwargs['n_estimators'] = 200
    self.model = RandomForestClassifier(**kwargs)

  def fit(self, X, y):
    self.model.fit(X, y)
    return self

  def predict(self, X):
    return self.model.predict(X)

  def predict_proba(self, X):
    return self.model.predict_proba(X)
