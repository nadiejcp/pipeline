class DataLoader:
  def __init__(self, config: dict):
    self.config = config

  def load(self):
    df = pd.read_csv(self.config["path"])
    date_format = self.config["date_format"]
    dates = pd.to_datetime(df[self.config["temporal"]], format=date_format)
    for feature in self.config["temporal_features"]:
      df[feature] = dates.dt[feature]
    df["hour"] = df["hour"] + df["minute"] / 60
    df.drop(columns=[self.config["temporal"], "minute"], inplace=True)
    return df

  def split(self, df: pd.DataFrame):
    df[self.config["target"]] = df[self.config["target"]].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
      df,
      test_size=self.config["split"]["test_size"],
      random_state=self.config["seed"],
      shuffle=True,
    )

    x_train, x_val, y_train, y_val = train_test_split(
      x_train,
      y_train,
      test_size=self.config["split"]["val_size"],
      random_state=self.config["seed"],
      shuffle=True,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test