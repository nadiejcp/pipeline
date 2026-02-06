import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
  def __init__(self, config: dict):
    self.config = config

  def load(self):
    df = pd.read_csv(self.config["data"]["path"]).dropna(subset=["Precipitation(in)"])
    date_format = self.config["features"]["date_format"]
    dates = pd.to_datetime(df[self.config["features"]["temporal"]], format=date_format)
    for feature in self.config["features"]["temporal_features"]:
      df[feature] = getattr(dates.dt, feature)
      df[feature] = df[feature].astype(float)
    df[self.config['features']["categorical"]] = (
      df[self.config['features']["categorical"]].astype(object).fillna("__missing__")
    )
    df["hour"] = df["hour"] + df["minute"] / 60
    df.drop(columns=[self.config["features"]["temporal"]], inplace=True)
    return df

  def split(self, df: pd.DataFrame):
    y = df[self.config["features"]["target"]].astype(int)
    feature_cols = (self.config["features"]["numerical"] + self.config["features"]["categorical"] + self.config["features"]["temporal_features"])
    X = df[feature_cols].drop(columns=['minute'])
    x_train, x_test, y_train, y_test = train_test_split(
      X,
      y,
      test_size=self.config["data"]["split"]["test_size"],
      random_state=self.config["seed"],
      shuffle=True,
      stratify=y
    )

    x_train, x_val, y_train, y_val = train_test_split(
      x_train,
      y_train,
      test_size=self.config["data"]["split"]["val_size"],
      random_state=self.config["seed"],
      shuffle=True,
      stratify=y_train
    )

    return x_train, x_val, x_test, y_train, y_val, y_test