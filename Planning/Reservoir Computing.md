---
tags:
  - machine-learning
  - dacss756
  - final-project
  - deep-learning
aliases:
  - Echo State Network
  - Echo State Networks
  - ESN
  - Liquid State Machine
  - Reservoir
---

**Reservoir computing** uses a large **recurrent network with fixed, untrained weights** (the reservoir) to turn an input time series into a rich, high-dimensional state, and trains only a simple **readout** (usually linear/ridge regression) from those states to the target. Echo state networks (Jaeger 2001; rate neurons) and liquid state machines (Maass, Natschläger & Markram 2002; spiking neurons) are the two original forms.

# The model (echo state network)
State update with leak rate $a$:
$$\mathbf{x}(t) = (1 - a)\,\mathbf{x}(t-1) + a\,\tanh\left(\mathbf{W}_{\text{in}}\mathbf{u}(t) + \mathbf{W}\mathbf{x}(t-1)\right)$$
$$\hat{\mathbf{y}}(t) = \mathbf{W}_{\text{out}}\,[1; \mathbf{u}(t); \mathbf{x}(t)]$$
- $\mathbf{u}(t)$: input at time $t$; $\mathbf{x}(t)$: reservoir state
- $\mathbf{W}_{\text{in}}$, $\mathbf{W}$: random (or, in a connectome reservoir, biological) and **fixed**
- $\mathbf{W}_{\text{out}}$: the only trained weights, fit by [[Ridge Regression]] on collected states

Because only a linear readout is trained, training is fast, convex and needs no backpropagation through time ([[Backpropagation]]).

# Key properties and hyperparameters
- **Echo state property**: the state must depend on the input history and forget the initial condition (fading memory). Usually ensured by scaling $\mathbf{W}$ so its **spectral radius** (largest absolute eigenvalue) is below about 1.
- **Spectral radius**: near 1 → long memory, rich dynamics ("edge of chaos"); smaller → shorter memory.
- **Leak rate** $a$: how fast states change; match it to the time scale of the input.
- **Input scaling**: how strongly inputs drive the reservoir vs its own dynamics.
- **Reservoir size and sparsity**: larger reservoirs give more features; sparse connectivity is typical.
- **Ridge penalty** on the readout ([[Regularization]]).
- **Washout**: discard the first states so the initial condition does not matter.

# Why it is useful
- The reservoir acts as a fixed non-linear **feature expansion with memory** ([[Feature Extraction]]); the readout picks out what predicts the target.
- Good for chaotic and nonlinear time series with modest data, and cheap to train many times (useful for comparing reservoirs).
- It tests the **computational value of a wiring diagram**: plug in a biological [[Connectome]] as $\mathbf{W}$ and compare with random networks of matched size, density and degree distribution (conn2res toolbox, Suárez et al. 2024).

# Cautions
- Performance depends heavily on the hyperparameters; tune them with temporal validation ([[Time-Series Cross-Validation]]).
- Compare against strong simple baselines (lagged ridge/logistic regression, gradient boosting).
- Random-network controls must match the connectome on the properties that matter (size, density, spectral radius, degree distribution) so differences can be attributed to wiring ([[Random Graph]]).

```python
import numpy as np
rng = np.random.default_rng(42)
N, a, rho = 500, 0.3, 0.9
W = rng.normal(size=(N, N)) * (rng.random((N, N)) < 0.05)        # sparse random reservoir
W *= rho / max(abs(np.linalg.eigvals(W)))                          # set spectral radius
W_in = rng.uniform(-0.5, 0.5, size=(N, U.shape[1]))
x, states = np.zeros(N), []
for u in U:                                                        # U: T x inputs
    x = (1 - a) * x + a * np.tanh(W_in @ u + W @ x); states.append(x.copy())
S = np.array(states)[washout:]
from sklearn.linear_model import RidgeCV
readout = RidgeCV(alphas=np.logspace(-6, 2, 20)).fit(S[:-h], y[washout + h:])   # h-step ahead
```

Related: [[Fly Connectome Conflict Forecasting]] | [[Neural Networks]] | [[Neuroevolution]]
