from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Preprocessor:
  def __init__(self, config):
    self.numeric_cols = config["numerical"]

    self.preprocessor = ColumnTransformer([
      ("num", StandardScaler(), self.numeric_cols),
      ("precip", Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler())
      ]), ["Precipitation(in)"]),
      ("cat", OneHotEncoder(), config["categorical"]),
    ])

  def fit(self, X):
    self.preprocessor.fit(X)

  def transform(self, X):
    return self.preprocessor.transform(X)
