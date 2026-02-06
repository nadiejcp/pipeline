from core.data_loader import DataLoader
from core.preprocessor import Preprocessor
from core.feature_engineer import FeatureEngineer
from models.factory import ModelFactory
from core.evaluator import Evaluator
from pathlib import Path
import joblib
import json
import torch
from tqdm import tqdm

TRAIN_PATH = Path("data/processed/train.joblib")
VAL_PATH   = Path("data/processed/val.joblib")
TEST_PATH  = Path("data/processed/test.joblib")

PREP_PATH  = Path("artifacts/preprocessor.joblib")
FE_PATH    = Path("artifacts/feature_engineer.joblib")

import numpy as np

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return super().default(obj)

class PipelineController:
  def __init__(self, config: dict):
    self.config = config

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {self.device}")

    self.data_loader = DataLoader(config)
    self.preprocessor = Preprocessor(config["features"])
    self.feature_engineer = FeatureEngineer(config["features"])
    self.model_factory = ModelFactory(config["models"], self.device)

  def load_processed_data(self, force_rebuild: bool = False):
    processed_exists = (
        TRAIN_PATH.exists() and
        VAL_PATH.exists() and
        TEST_PATH.exists() and
        PREP_PATH.exists()
    )
    if not force_rebuild and processed_exists:
      print("Loading processed data...")

      X_train, y_train = joblib.load(TRAIN_PATH)
      X_val, y_val     = joblib.load(VAL_PATH)
      X_test, y_test   = joblib.load(TEST_PATH)

      self.preprocessor = joblib.load(PREP_PATH)
      self.feature_engineer = joblib.load(FE_PATH)

      return X_train, X_val, X_test, y_train, y_val, y_test

    print("Loading raw data...")
    df = self.data_loader.load()
    print('Data loaded')
    X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split(df)
    print('Data splitted')

    X_train = self.feature_engineer.transform(X_train)
    X_val = self.feature_engineer.transform(X_val)
    X_test = self.feature_engineer.transform(X_test)
    print('Feature engineered')

    joblib.dump(self.feature_engineer, "artifacts/feature_engineer.joblib")

    self.preprocessor.fit(X_train)
    X_train = self.preprocessor.transform(X_train)
    X_val = self.preprocessor.transform(X_val)
    X_test = self.preprocessor.transform(X_test)
    print('Preprocessor fitted')

    joblib.dump(self.preprocessor, "artifacts/preprocessor.joblib")

    joblib.dump((X_train, y_train), "data/processed/train.joblib")
    joblib.dump((X_val, y_val), "data/processed/val.joblib")
    joblib.dump((X_test, y_test), "data/processed/test.joblib")
    print('Data processed')

    return X_train, X_val, X_test, y_train, y_val, y_test

  def save_model(self, model, model_name: str):
    path = Path(f"artifacts/models/{model_name}.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

  def save_metrics(self, metrics: dict, model_name: str, split="val"):
    path = Path(f"artifacts/metrics/{model_name}_{split}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
      json.dump(metrics, f, indent=2, cls=NumpyEncoder)

  def run(self, force_rebuild: bool = False):
    X_train, X_val, X_test, y_train, y_val, y_test = self.load_processed_data(force_rebuild)
    results = {}
    print(f'Training models...')
    for model_name in tqdm(self.model_factory.list_models(), desc="Pipeline"):
      model_path = Path(f"artifacts/models/{model_name}.joblib")
      if model_path.exists() and not force_rebuild:
        print(f"Loading existing model: {model_name}...")
        model = joblib.load(model_path)
        if hasattr(model, 'torch_model') and model.torch_model is not None:
          model.device = self.device
          model.torch_model = model.torch_model.to(self.device)
      else:
        model = self.model_factory.create(model_name)
        print(f'Model {model_name} created')
        print(f'Training {model_name}...')
        model.fit(X_train, y_train)
        print(f'Model {model_name} trained')
        self.save_model(model, model_name)
        print(f'Model {model_name} saved')

      print(f'Evaluating {model_name}...')
      self.evaluator = Evaluator(model)
      metrics = self.evaluator.evaluate(X_val, y_val)
      print(f'Metrics for {model_name}: {metrics}')
      self.save_metrics(metrics, model_name)
      metrics = self.evaluator.evaluate(X_test, y_test)
      print(f'Metrics for {model_name}: {metrics}')
      self.save_metrics(metrics, model_name, split="test")

      results[model_name] = metrics
    return results
