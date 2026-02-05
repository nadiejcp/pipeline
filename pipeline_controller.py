from pipeline.core.data_loader import DataLoader
from pipeline.core.preprocessor import Preprocessor
from pipeline.core.feature_engineer import FeatureEngineer
from pipeline.core.evaluator import Evaluator
from pipeline.models.factory import ModelFactory
from pipeline.utils.artifacts import ArtifactManager
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

    # Core components
    self.data_loader = DataLoader(config["data"])
    self.preprocessor = Preprocessor(config["features"])
    self.feature_engineer = FeatureEngineer(config["features"])
    self.evaluator = Evaluator(config["metrics"])

    # Utilities
    self.model_factory = ModelFactory(config["models"])
    self.artifact_manager = ArtifactManager(config["output"])

  def run(self):
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

    results = {}

    # 5. Train & evaluate each model
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
