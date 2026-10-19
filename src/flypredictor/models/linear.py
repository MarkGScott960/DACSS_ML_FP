"""Penalised logistic regression (plan 6.2). Ridge is the confirmatory primary model (D-20b); elastic net and lasso are in the lineup.

``ElasticNetLogit`` wraps the training-only ``FeaturePreprocessor`` and scikit-learn's
``LogisticRegression(penalty='elasticnet', solver='saga')``. No class reweighting (plan 6.5).
The hyperparameter grid and the tuning loop come with the model zip; the power simulation
uses fixed values.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from ..data.features import FeaturePreprocessor, column_groups, coverage_rule_groups


class ElasticNetLogit:
    def __init__(self, cfg: dict[str, Any], columns: list[str], C: float = 0.1, l1_ratio: float = 0.5,
                 max_iter: int = 3000, tol: float = 1e-4, seed: int = 0):
        self.cfg, self.columns = cfg, list(columns)
        self.C, self.l1_ratio, self.max_iter, self.tol, self.seed = C, l1_ratio, max_iter, tol, seed

    def fit(self, X: pd.DataFrame, y) -> "ElasticNetLogit":
        self.pre_ = FeaturePreprocessor(self.columns, column_groups(self.cfg), coverage_rule_groups(self.cfg))
        Z = self.pre_.fit_transform(X)
        kw = dict(penalty="elasticnet", C=self.C, solver="saga", l1_ratio=self.l1_ratio,
                  max_iter=self.max_iter, tol=self.tol, random_state=self.seed)
        with warnings.catch_warnings():
            # scikit-learn 1.8 deprecates `penalty` (still honoured); keep it explicit for older versions
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.m_ = LogisticRegression(**kw).fit(Z, np.asarray(y, int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.m_.predict_proba(self.pre_.transform(X))[:, 1]

    @property
    def coef_(self) -> pd.Series:
        return pd.Series(self.m_.coef_[0], index=self.pre_.output_names)


class RidgeLogit(ElasticNetLogit):
    """L2-penalised logit (lbfgs). Part of the plan-6.2 lineup, and the fast stand-in for the
    elastic net in the power simulation (about 300x faster; checked against it in notebook 04)."""

    def __init__(self, cfg: dict[str, Any], columns: list[str], C: float = 0.05, max_iter: int = 2000, **_: Any):
        super().__init__(cfg, columns, C=C, max_iter=max_iter)

    def fit(self, X: pd.DataFrame, y) -> "RidgeLogit":
        self.pre_ = FeaturePreprocessor(self.columns, column_groups(self.cfg), coverage_rule_groups(self.cfg))
        Z = self.pre_.fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.m_ = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(Z, np.asarray(y, int))
        return self
