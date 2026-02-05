import pandas as pd

class FeatureEngineer:
  def __init__(self, config):
    self.temporal_features = config["temporal_features"]
    self.periods = {"hour": 24, "month": 12, "dayofweek": 7}

  def transform(self, df: pd.DataFrame):
    df = df.copy()

    for feature in self.temporal_features:
      period = self.periods[feature]
      df[f"{feature}_sin"] = np.sin(2 * np.pi * df[feature] / period)
      df[f"{feature}_cos"] = np.cos(2 * np.pi * df[feature] / period)

    df.drop(columns=self.temporal_features, inplace=True, errors="ignore")
    return df
