# The Optimization Layer — Gaussian-Process Bayesian Optimization with Entropy Search for LQG Hyperparameter Tuning

> **Scope.** This article specifies *only* the optimization layer: the surrogate model, the hyperparameter-learning inner loop, and the Entropy-Search acquisition. It is a mathematical and algorithmic specification, not an implementation. The layer below (synthesising and simulating an LQR + Kalman controller and scoring it) is treated as an opaque oracle. The reader is the agent that will construct this algorithm.

---

## 1. The Optimization Problem

We are tuning the LQG design weights collected into a single vector

$$

\boldsymbol{\theta} = \mathrm{vec}(Q, R, W, V) \in \Theta \subset \mathbb{R}^{d},

$$

where $Q, R$ are the LQR cost weights and $W, V$ the noise covariances the Kalman filter assumes. The domain $\Theta$ is bounded; each parameter is searched in **log-space**, because lengthscales, variances, and covariance diagonals are positive and span orders of magnitude. Optimising the logarithm turns a positivity-constrained problem into an unconstrained box and equalises the scales the kernel must model.

The objective is the realised closed-loop cost,

$$

J(\boldsymbol{\theta}) = \mathbb{E}\!\left[\, \text{closed-loop quadratic cost of the controller built from } \boldsymbol{\theta} \,\right],

$$

which has three defining properties: it is **expensive** (each query is a full simulated rollout), **noisy** (each rollout realises random process/measurement noise and a random initial condition), and **black-box** (no gradient, no analytic form — only point evaluations). The optimizer interacts with the rest of the system through a single abstraction: an oracle that maps a candidate $\boldsymbol{\theta}$ to a scalar observation

$$

y = J(\boldsymbol{\theta}) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma_{n}^{2}).

$$

The homoscedastic Gaussian-noise assumption is a deliberate surrogate: the true rollout cost is a generalised chi-squared variable, but averaging $n_{r}$ independent rollouts and invoking the central limit theorem drives the residual toward a Normal with variance $\mathcal{O}(1/n_{r})$. The variance $\sigma_{n}^{2}$ is not fixed by hand; it is learned as a model hyperparameter (Section 3).

**The goal is the minimiser, not the trajectory of low values.** We want to identify

$$

\boldsymbol{\theta}^{\star} = \arg\min_{\boldsymbol{\theta} \in \Theta} J(\boldsymbol{\theta})

$$

in as few oracle calls as possible. This framing — inference about the *location* of the optimum — is what makes Entropy Search the correct acquisition, as opposed to value-greedy rules that merely chase low observed costs.

---

## 2. The Gaussian-Process Surrogate

The optimizer maintains a probabilistic model of the cost surface over $\Theta$, a Gaussian process

$$

J \sim \mathcal{GP}(0, k),

$$

with a smooth stationary kernel — squared-exponential or Matérn, with **automatic relevance determination** (one lengthscale per dimension), so the model can learn that some weights matter far more than others. A zero mean prior is used after the targets are standardised (Section 6).

Given a dataset of $t$ observations $\mathcal{D}_{t} = \{(\boldsymbol{\theta}_{i}, y_{i})\}_{i=1}^{t}$, write $\Theta_{t}$ for the stacked inputs, $\mathbf{y}_{t}$ for the targets, $K_{t}$ for the Gram matrix with $[K_{t}]_{ij} = k(\boldsymbol{\theta}_{i}, \boldsymbol{\theta}_{j})$, and $\mathbf{k}_{t}(\boldsymbol{\theta})$ for the vector of cross-covariances between a query point and the data. The posterior at any $\boldsymbol{\theta}$ is Gaussian, $J(\boldsymbol{\theta}) \mid \mathcal{D}_{t} \sim \mathcal{N}(\mu_{t}(\boldsymbol{\theta}), \sigma_{t}^{2}(\boldsymbol{\theta}))$, with

$$

\mu_{t}(\boldsymbol{\theta}) = \mathbf{k}_{t}(\boldsymbol{\theta})^{\top}\left[K_{t} + \sigma_{n}^{2} I\right]^{-1} \mathbf{y}_{t},

$$

$$

\sigma_{t}^{2}(\boldsymbol{\theta}) = k(\boldsymbol{\theta}, \boldsymbol{\theta}) - \mathbf{k}_{t}(\boldsymbol{\theta})^{\top}\left[K_{t} + \sigma_{n}^{2} I\right]^{-1} \mathbf{k}_{t}(\boldsymbol{\theta}).

$$

Two structural facts drive everything downstream. The posterior **variance does not depend on the observed targets**, only on where data has been collected — so the model can predict, before paying for an evaluation, exactly how much an observation at $\boldsymbol{\theta}$ would shrink its uncertainty. And the posterior is the **complete state of belief**: the acquisition is a deterministic functional of the pair $(\mu_{t}, \sigma_{t}^{2})$. The factored system $[K_{t} + \sigma_{n}^{2} I]^{-1}$ should be formed once per model state (via a symmetric positive-definite factorisation with diagonal jitter) and reused for all queries.

---

## 3. The Inner Loop — Learning the GP Hyperparameters (ML-II)

The kernel itself carries free hyperparameters

$$

\boldsymbol{\phi} = (\ell_{1}, \ldots, \ell_{d}, \sigma_{f}, \sigma_{n}),

$$

the per-dimension lengthscales, the signal amplitude, and the observation-noise level. These are not guessed; they are fit by **type-II maximum likelihood** (ML-II): maximising the log marginal likelihood of the data under the model,

$$

\log p(\mathbf{y}_{t} \mid \Theta_{t}, \boldsymbol{\phi}) = -\tfrac{1}{2}\, \mathbf{y}_{t}^{\top}\left[K_{t} + \sigma_{n}^{2} I\right]^{-1} \mathbf{y}_{t} - \tfrac{1}{2}\log\left\lvert K_{t} + \sigma_{n}^{2} I\right\rvert - \tfrac{t}{2}\log 2\pi.

$$

The three terms are a data-fit reward, a complexity penalty (the log-determinant), and a constant; their balance is the automatic Occam's razor that prevents the surrogate from over- or under-smoothing. Maximisation is performed by gradient ascent in **log-space** (the hyperparameters are positive), and because the objective is **non-convex**, from **multiple random restarts**, keeping the best marginal likelihood.

This inner loop runs **every time a new observation is appended** to $\mathcal{D}_{t}$, before the acquisition is evaluated. The nesting is strict and must not be confused: the *outer* loop optimises the controller weights $\boldsymbol{\theta}$ over the true cost; the *inner* loop optimises the kernel hyperparameters $\boldsymbol{\phi}$ over the marginal likelihood of the surrogate. They have different parameters, different objectives, and different optimisers.

---

## 4. Entropy Search — the Acquisition Function

### 4.1 The quantity of interest is the belief over the minimiser

Classical acquisitions (Expected Improvement, Upper Confidence Bound) are *local utilities*: they score a candidate by how good its predicted value is. Entropy Search abandons this. It maintains an explicit belief over the **location of the optimum**,

$$

p_{\min}(\boldsymbol{\theta}) = \Pr\!\left[\, \boldsymbol{\theta} = \arg\min_{\boldsymbol{\theta}' \in \Theta} J(\boldsymbol{\theta}') \,\right] = \int p(J)\, \prod_{\boldsymbol{\theta}' \neq \boldsymbol{\theta}} \mathbb{1}\!\left[\, J(\boldsymbol{\theta}') \geq J(\boldsymbol{\theta}) \,\right] \, dJ,

$$

a distribution induced by the GP posterior. The indicator product is a logical gate: it is $1$ for exactly those sample paths $J$ minimised at $\boldsymbol{\theta}$, so the integral accumulates the posterior mass of all cost surfaces whose minimum sits at $\boldsymbol{\theta}$. Our ignorance about the optimum is the Shannon entropy of this belief,

$$

H[p_{\min}] = -\sum_{i} p_{\min}(\boldsymbol{\theta}_{i}) \log p_{\min}(\boldsymbol{\theta}_{i}),

$$

high when many configurations are plausibly optimal, low when the belief concentrates on one. **Optimisation is reframed as the monotone destruction of this entropy.**

### 4.2 The acquisition is expected information gain

The next evaluation is placed where the observation is expected to reduce $H[p_{\min}]$ the most. The expected reduction is, exactly, the **mutual information** between the minimiser and the would-be observation:

$$

\alpha_{\mathrm{ES}}(\boldsymbol{\theta}) = H[p_{\min}] - \mathbb{E}_{y_{\boldsymbol{\theta}}}\!\left[\, H[\, p_{\min} \mid y_{\boldsymbol{\theta}} \,]\, \right] = I\!\left(\boldsymbol{\theta}^{\star};\, y_{\boldsymbol{\theta}} \mid \mathcal{D}_{t}\right),

$$

and the next query is $\boldsymbol{\theta}_{t+1} = \arg\max_{\boldsymbol{\theta}} \alpha_{\mathrm{ES}}(\boldsymbol{\theta})$. The first term is constant in $\boldsymbol{\theta}$, so the rule selects the candidate whose observation is expected to leave the **smallest residual entropy** over the location of the optimum. Critically, this is a *nonlocal* criterion: an evaluation perturbs $p_{\min}$ everywhere, so a candidate that is not itself promising can win if probing it most sharply collapses the belief — exactly the behaviour wanted when each rollout is precious.

### 4.3 Why $p_{\min}$ is intractable, and how to compute the acquisition

The belief $p_{\min}$ has no closed form: the integral is over an infinite-dimensional function space, and even on a finite set of locations it is a Gaussian integral over a polyhedral cone (the region carved by the inequalities $J(\boldsymbol{\theta}') \geq J(\boldsymbol{\theta})$), which has no analytic solution. The acquisition needs $p_{\min}$ *twice* — before and after a hypothetical observation, averaged over its unknown value — so two layers of approximation are required.

The algorithm the agent must build estimates $\alpha_{\mathrm{ES}}$ as follows.

**Discretisation.** Approximate $\Theta$ by a finite set of **representer points** $\mathcal{R} = \{\boldsymbol{\theta}_{r}\}_{r=1}^{N}$. These need not lie on a grid (a grid suffers the curse of dimensionality); they are drawn from a non-uniform proposal measure — for instance Expected Improvement or Probability of Improvement renormalised as a sampling density — which concentrates resolution where $p_{\min}$ has mass. The candidate $\boldsymbol{\theta}$ being scored is included in $\mathcal{R}$. A poor proposal only costs more representers; in the limit $N \to \infty$ any full-support proposal is equivalent.

**Estimating $p_{\min}$ on $\mathcal{R}$.** Two routes:
- *Monte Carlo over GP samples.* Draw $S$ joint posterior samples of the cost vector over $\mathcal{R}$; take the $\arg\min$ of each sample; $p_{\min}(\boldsymbol{\theta}_{r})$ is the empirical fraction of samples minimised at $\boldsymbol{\theta}_{r}$. This is asymptotically exact and simple, but non-differentiable.
- *Expectation Propagation.* Treat the Gaussian belief as a prior message and each inequality $J(\boldsymbol{\theta}_{r'}) \geq J(\boldsymbol{\theta}_{r})$ as a one-sided likelihood factor, and run moment-matching to convergence; the per-point normalisers approximate $p_{\min}$. This is more expensive but returns $p_{\min}$ as a *differentiable* function of $(\mu_{t}, \sigma_{t}^{2})$, enabling gradient-based maximisation of the acquisition.

**Marginalising the hypothetical outcome.** For the candidate $\boldsymbol{\theta}$, the unobserved outcome is $y_{\boldsymbol{\theta}} \sim \mathcal{N}(\mu_{t}(\boldsymbol{\theta}),\, \sigma_{t}^{2}(\boldsymbol{\theta}) + \sigma_{n}^{2})$. Average the post-observation entropy over this distribution using a small fixed set of innovation samples (or quadrature nodes), drawn **once and reused** across the optimisation of the acquisition so that $\alpha_{\mathrm{ES}}$ is a smooth function of $\boldsymbol{\theta}$. For each sampled outcome, apply a rank-one *fantasy update* to the GP posterior (the predictive change to the covariance is deterministic; only the mean shift carries the random innovation), recompute $p_{\min}$ on $\mathcal{R}$, and take its entropy.

### 4.4 A closed-form variant: value-based information gain

If the full $p_{\min}$ machinery is too costly, the information-theoretic goal can be preserved while targeting the **optimal value** $J^{\star} = \min_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ instead of its location. Because $J^{\star}$ is scalar, the symmetry of mutual information yields a closed form. Swapping the roles of $J^{\star}$ and the observation,

$$

\alpha_{\mathrm{MES}}(\boldsymbol{\theta}) = I\!\left(J^{\star};\, y_{\boldsymbol{\theta}} \mid \mathcal{D}_{t}\right) = H\!\left[\, p(y_{\boldsymbol{\theta}} \mid \mathcal{D}_{t}) \,\right] - \mathbb{E}_{J^{\star}}\!\left[\, H\!\left[\, p(y_{\boldsymbol{\theta}} \mid \mathcal{D}_{t}, J^{\star}) \,\right] \,\right].

$$

The first term is the entropy of a one-dimensional Gaussian. For the second, conditioning on $J^{\star}$ being the minimum forces $J(\boldsymbol{\theta}) \geq J^{\star}$, which **truncates the predictive Gaussian from below** at $J^{\star}$. With $\psi$ and $\Psi$ the standard-normal density and cumulative functions (named to avoid the kernel hyperparameter $\boldsymbol{\phi}$), and the standardised gap

$$

\gamma_{J^{\star}}(\boldsymbol{\theta}) = \frac{\mu_{t}(\boldsymbol{\theta}) - J^{\star}}{\sigma_{t}(\boldsymbol{\theta})},

$$

the per-sample information gain has the closed form below; averaging over minimum-values $J^{\star}$ sampled from the GP posterior (via a Gumbel approximation, or by sampling functions and minimising them) gives

$$

\alpha_{\mathrm{MES}}(\boldsymbol{\theta}) \approx \frac{1}{\lvert \mathcal{F} \rvert} \sum_{J^{\star} \in \mathcal{F}} \left[\, \frac{\gamma_{J^{\star}}(\boldsymbol{\theta})\, \psi\!\left(\gamma_{J^{\star}}(\boldsymbol{\theta})\right)}{2\, \Psi\!\left(\gamma_{J^{\star}}(\boldsymbol{\theta})\right)} - \log \Psi\!\left(\gamma_{J^{\star}}(\boldsymbol{\theta})\right) \,\right].

$$

Each term needs only the posterior mean and variance at $\boldsymbol{\theta}$ and a handful of sampled optima — no representer set, no fantasy updates, no EP. The trade is explicit: the location-based $\alpha_{\mathrm{ES}}$ is the faithful realisation of the belief-over-the-minimiser goal and is the project target; the value-based $\alpha_{\mathrm{MES}}$ is a cheap, closed-form approximation of the same information principle, suitable as a fallback or a baseline.

### 4.5 Maximising the acquisition

Whichever estimator is used, $\alpha(\boldsymbol{\theta})$ is itself optimised over $\Theta$ at each outer step — a cheap, repeatable, analytic optimisation (unlike the expensive true objective). A standard scheme is to evaluate $\alpha$ on a moderate candidate set (random or proposal-drawn), then refine the best candidates locally. The maximiser becomes the next controller configuration sent to the oracle.

---

## 5. The Integrated Outer Loop

```pseudo
\begin{algorithm}
\caption{GP-BO with Entropy Search for LQG Tuning}
\begin{algorithmic}
\Procedure{Optimize}{$\Theta, T, n_{\text{init}}, n_{r}$}
    \State sample $n_{\text{init}}$ initial $\boldsymbol{\theta}$ from $\Theta$; query the oracle ($n_{r}$ averaged rollouts each)
    \State standardise targets; fit GP hyperparameters $\boldsymbol{\phi}$ by ML-II \Comment{Section 3}
    \For{$t = n_{\text{init}}+1, \ldots, T$}
        \State condition the GP on $\mathcal{D}_{t-1}$ to obtain $\mu_{t-1}, \sigma_{t-1}^{2}$ \Comment{Section 2}
        \State draw representer points $\mathcal{R}$ from a proposal measure (ES route)
        \State or sample minimum-values $\mathcal{F}$ from the posterior (MES route)
        \State $\boldsymbol{\theta}_{t} \gets \arg\max_{\boldsymbol{\theta} \in \Theta} \alpha(\boldsymbol{\theta})$ \Comment{Section 4}
        \State $y_{t} \gets$ oracle$(\boldsymbol{\theta}_{t})$, averaged over $n_{r}$ rollouts
        \State append $(\boldsymbol{\theta}_{t}, y_{t})$ to the dataset; re-standardise
        \State refit $\boldsymbol{\phi}$ by ML-II \Comment{inner loop runs every step}
    \EndFor
    \State \Return $\arg\min_{\boldsymbol{\theta}} \mu_{T}(\boldsymbol{\theta})$ \Comment{report the posterior-mean minimiser}
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

The returned answer is the minimiser of the **posterior mean**, not the best observed point — under noise, the lowest observation may be a lucky draw, whereas the posterior mean integrates the evidence.

---

## 6. Numerical and Statistical Discipline Specific to This Layer

A handful of practices protect the surrogate and the search.

**Target standardisation.** Before fitting, transform the observed costs to zero mean and unit variance, and store the transform so predictions map back. The kernel amplitude and noise are otherwise unidentifiable against arbitrary cost scales.

**Finite cost surface.** The oracle must return a finite scalar. A diverged or unstable controller is mapped to a large *finite* penalty, never to infinity or NaN — a single non-finite target poisons the entire GP fit.

**Conditioning.** The kernel matrix is near-singular when representers or data cluster. Symmetrise before factoring and add diagonal jitter (the learned noise $\sigma_{n}^{2}$ plus a small floor); escalate jitter if factorisation fails, and treat a large negative predictive variance as a bug rather than round-off (clip small negatives to zero).

**Determinism.** Every stochastic draw inside the optimizer — initial designs, proposal sampling, the GP function samples in Entropy Search, the innovation samples in the fantasy updates — flows from a single seeded generator threaded through the call signatures. The acquisition's sample sets are drawn once per state and reused, both for smoothness and for reproducibility.

**Exploration safeguard.** Because data is selected by the model's own belief, the search can over-concentrate. Hyperpriors on the kernel parameters and an occasional random exploratory query both mitigate belief-driven collapse.

---

## 7. Two Distinctions That Must Not Be Blurred

This layer hosts two pairs of easily-confused objects.

The **two nested optimisations** (Sections 1 and 3): the outer loop searches controller weights $\boldsymbol{\theta} = \mathrm{vec}(Q,R,W,V)$ over the true cost; the inner loop searches kernel hyperparameters $\boldsymbol{\phi} = (\ell, \sigma_{f}, \sigma_{n})$ over the marginal likelihood. They never share an optimiser or an objective.

The **two entropies**: the information-theoretic entropy $H[p_{\min}]$ of the belief over the optimum, defined in Section 4, is the engine of the acquisition and lives only here. Any *physical* trajectory-entropy metric of a controller's response is a different quantity entirely and belongs to the scoring layer, never to this one.