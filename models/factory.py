from .baseline import BaselineModel
from .random_forest import RandomForestModel
from .mlp import MLPModel

class ModelFactory:
  def __init__(self, config: dict, device=None):
    self.config = config
    self.device = device

  def list_models(self):
    """Returns the list of model names defined in the configuration."""
    return list(self.config.keys())

  def create(self, name: str):
    """Creates and returns a model instance based on the configuration name."""
    if name not in self.config:
      raise ValueError(f"Model configuration '{name}' not found in config.")

    model_conf = self.config[name].copy()
    model_type = model_conf.pop("type")

    if model_type == "logistic_regression":
      return BaselineModel(**model_conf, device=self.device)
    elif model_type == "random_forest":
      return RandomForestModel(**model_conf, device=self.device)
    elif model_type == "mlp":
      return MLPModel(**model_conf, device=self.device)
    else:
      raise ValueError(f"Unknown model type: {model_type}")
