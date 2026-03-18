import joblib
from typing import Any, Dict

def load_model(model_path: str = "models/risk_model.pkl") -> Dict[str, Any]:
    """
    Loads model artifact saved by the notebook.

    Expected structure:
      {
        "model": trained sklearn model or pipeline,
        "features": list[str],
        "metrics": dict,
        "model_name": str (optional)
      }
    """
    return joblib.load(model_path)