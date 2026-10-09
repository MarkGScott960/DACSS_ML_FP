"""Models (written only after the pre-registration is frozen; plan section 6).

  baselines.py   persistence bins, EDI-momentum logit
  linear.py      lasso / ridge / elastic net (primary confirmatory model), regime interactions
  gbm.py         gradient boosting (theory-driven and kitchen-sink)
  gru.py         GRU over country / episode histories
  survival.py    random survival forest, gradient-boosted Cox, Cox (stage 2)
  stacking.py    logistic meta-learner over out-of-fold predictions
  reservoir/     optional exploratory reservoir row (fly, matched random, leaky bank)
"""
