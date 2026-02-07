from sklearn.metrics import (
  accuracy_score,
  f1_score,
  confusion_matrix,
  classification_report
)

class Evaluator:
  def __init__(self, model):
    self.model = model

  def evaluate(self, X, y, split_name="val"):
    y_pred = self.model.predict(X)

    results = {
      "split": split_name,
      "accuracy": accuracy_score(y, y_pred),
      "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
      "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
      "confusion_matrix": confusion_matrix(y, y_pred),
      "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
    }

    return results
