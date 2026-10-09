"""Pre-freeze audits (plan section 7.2). Outcome-only or feature-only: no audit here
looks at a relationship between a predictor and an outcome.

coverage   first/last year and non-missing share per variable; missingness by decade;
           missingness flags that act as a proxy for time (era-proxy check)
maturity   episode recount, start->confirmation and breakdown->confirmation lags (sets buffers)
realtime   agreement between state sources (ERT re-runs vs rule vs hindsight)
definitions RoW vs ERT regime type in the risk sets; RoW transitions ERT does not count as breakdowns
"""
