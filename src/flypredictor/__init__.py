"""FlyPredictor: forecasting democratic backsliding with V-Dem and ERT.

DACSS 756 final project (Mark Scott, UMass Amherst). The plan of record is
docs/plan_of_record.md; every analytic choice lives in config/*.yaml.

Sub-packages
------------
data       panel, democratic episodes, labels (stage 1 onset/confirmation, stage 2 breakdown)
audit      outcome-only checks that feed the pre-registration (coverage, missingness, maturity)
eval       splits, metrics, bootstrap, prediction store, lockbox            (later zip)
models     baselines, linear, boosting, GRU, survival, stacking             (after pre-registration)
interpret  drop-group refits, permutation importance, SHAP                  (after pre-registration)
"""

from .config import load_config
from .paths import Paths

__all__ = ["load_config", "Paths"]
__version__ = "0.1.0"
