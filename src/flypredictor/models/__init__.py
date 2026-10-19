"""Models (plan section 6).

Built in zip 2 (needed by the power simulation; no model has been fit to real outcomes):
  baselines.py   persistence bins, EDI-momentum logit
  linear.py      ridge logit (primary confirmatory model, D-20b) and elastic net

After the pre-registration is frozen:
  gbm.py         gradient boosting (theory-driven and kitchen-sink)
  gru.py         GRU over country / episode histories
  survival.py    random survival forest, gradient-boosted Cox, Cox (stage 2, exploratory)
  stacking.py    logistic meta-learner over out-of-fold predictions
  reservoir/     optional exploratory reservoir row (fly, matched random, leaky bank)
"""
