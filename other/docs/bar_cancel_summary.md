# Bar Cancel Summary (Global vs EMA)

Global (full-frame mean) bar cancel
```
Given frames f_t(x,y), t=1..N:
μ(x,y) = (1/N) ∑_{t=1}^N f_t(x,y)
H_x(μ) = horizontal blur of μ along x (1D conv with kernel k)
b(x,y) = μ(x,y) − H_x(μ)(x,y)
g_t(x,y) = clip( f_t(x,y) − β · b(x,y), 0, 1 )
```

EMA (online) bar cancel
```
Initialize μ_0 = f_0 (or 0). For each frame t:
μ_t = (1−α) μ_{t−1} + α f_t
(optional bias-correct) μ̂_t = μ_t / (1 − (1−α)^t)
b_t = μ̂_t − H_x(μ̂_t)      (or use μ_t if no correction)
g_t = clip( f_t − β · b_t, 0, 1 )
```

Definitions:
- α: EMA smoothing factor (0<α≤1)
- β: cancel strength (cancel_alpha)
- H_x: horizontal blur (stripe low-pass)
- clip: clamp to display range (e.g., [0,1])
