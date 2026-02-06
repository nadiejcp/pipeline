from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

class Preprocessor:
  def __init__(self, config):
    numerical = [c for c in config["numerical"] if c != "Precipitation(in)"]
    self.preprocessor = ColumnTransformer([
      ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
        ]), numerical),
      ("precip", Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler())
      ]), ["Precipitation(in)"]),
      ("cat", Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
      ]), config["categorical"]),
    ], remainder="drop")

  def fit(self, X):
    self.preprocessor.fit(X)

  def transform(self, X):
    return self.preprocessor.transform(X)
