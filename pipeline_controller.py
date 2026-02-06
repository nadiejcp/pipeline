from core.data_loader import DataLoader
from core.preprocessor import Preprocessor
from core.feature_engineer import FeatureEngineer
from core.evaluator import Evaluator
from models.factory import ModelFactory
from utils.artifacts import ArtifactManager
from pathlib import Path
import joblib

TRAIN_PATH = Path("data/processed/train.joblib")
VAL_PATH   = Path("data/processed/val.joblib")
TEST_PATH  = Path("data/processed/test.joblib")

PREP_PATH  = Path("artifacts/preprocessor.joblib")
FE_PATH    = Path("artifacts/feature_engineer.joblib")

class PipelineController:
  def __init__(self, config: dict):
    self.config = config

    self.data_loader = DataLoader(config["data"])
    self.preprocessor = Preprocessor(config["features"])
    self.feature_engineer = FeatureEngineer(config["features"])
    self.evaluator = Evaluator(config["metrics"])

    self.model_factory = ModelFactory(config["models"])
    self.artifact_manager = ArtifactManager(config["output"])

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
    X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split(df)

    X_train = self.feature_engineer.transform(X_train)
    X_val = self.feature_engineer.transform(X_val)
    X_test = self.feature_engineer.transform(X_test)
    joblib.dump(self.feature_engineer, "artifacts/feature_engineer.joblib")

    self.preprocessor.fit(X_train)
    X_train = self.preprocessor.transform(X_train)
    X_val = self.preprocessor.transform(X_val)
    X_test = self.preprocessor.transform(X_test)

    joblib.dump(self.preprocessor, "artifacts/preprocessor.joblib")

    joblib.dump((X_train, y_train), "data/processed/train.joblib")
    joblib.dump((X_val, y_val), "data/processed/val.joblib")
    joblib.dump((X_test, y_test), "data/processed/test.joblib")

    return X_train, X_val, X_test, y_train, y_val, y_test

  def run(self, force_rebuild: bool = False):
    X_train, X_val, X_test, y_train, y_val, y_test = self.load_processed_data(force_rebuild)
    print("Training models...")
    results = {}
    for model_name in self.model_factory.list_models():
      model = self.model_factory.create(model_name)

      model.fit(X_train, y_train)

      metrics = self.evaluator.evaluate(
        model=model,
        X_val=X_val,
        y_val=y_val
      )

      # 6. Save artifacts
      self.artifact_manager.save_model(model, model_name)
      self.artifact_manager.save_metrics(metrics, model_name)

      results[model_name] = metrics

    # 7. Compare models
    self.artifact_manager.save_summary(results)

    return results
