import jax.numpy as jnp
import jax.scipy as jscipy
from jax import random, vmap, jit
from jax.scipy.special import logit
from jax.nn import sigmoid

# ============================================================
# Two-group SIR monitoring model (SDE -> Euler–Maruyama)
# State  : x = [S1, I1, S2, I2]   (R is implicit via N - S - I)
# Params : theta = [beta1, gamma1, beta2, gamma2]
# Design : xi_t is a value in R, affects observation only
# Obs    : y_t^1 ~ Poisson(kappa  * rho_1 * (I_1 / N_1) * (1.0 / (1.0 + jnp.exp(-xi_t)))))
#          y_t^2 ~ Poisson(kappa  * rho_2 * (I_2 / N_2) * (1.0 / (1.0 + jnp.exp(-(1-xi_t)))))
# Drift  : f(x) = S a(x), with a(x) the rates of the 4 reactions
# Diffusion: state-dependent, cov(x) = S diag(a(x)) S^T * dt (with small jitter)
# ============================================================

# -----------------------------
# Epidemic hyperparameters (can be overridden)
# -----------------------------
_N = jnp.array([200.0, 200.0])               # population sizes (N1, N2)
_I0 = jnp.array([5.0, 5.0])                  # initial infectious counts (I1_0, I2_0)
_RHO = jnp.array([0.95, 0.5])                    # detection scales (rho1, rho2)
_KAPPA = 100.0                                   # sampling effort
_MIX = jnp.array([[0.9, 0.1],                   # mixing matrix M (2x2)
                  [0.1, 0.9]])
_DESIGN_DIM = 1  # design dimension (1 for xi in R, 2 for (xi1, xi2) on simplex)
_STATE_DIM = 4   # state dimension (S1, I1, S2, I2)
_PARAM_DIM = 4   # parameter dimension (beta1, gamma1, beta2, gamma2)
_OBS_DIM = 2     # observation dimension (y1, y2)


def set_epidemic_hyperparams(N=None, rho=None, kappa=None, M=None):
    """
    Configure population sizes, detection scales, sampling effort, and mixing matrix.
    N     : array (2,) of group sizes
    rho   : array (2,) of detection scales
    kappa : float > 0
    M     : array (2,2) mixing matrix, rows sum not required; model uses M @ (I/N)
    """
    global _N, _RHO, _KAPPA, _MIX
    if N is not None:
        _N = jnp.array(N, dtype=float)
    if rho is not None:
        _RHO = jnp.array(rho, dtype=float)
    if kappa is not None:
        _KAPPA = float(kappa)
    if M is not None:
        _MIX = jnp.array(M, dtype=float)


# -----------------------------
# Helpers: rates, stoichiometry, drift & diffusion
# -----------------------------

def _force_of_infection(I, N, M):
    return (M @ (I / N))

def _rates(state, param):
    """Compute reaction rates a(x) = (lambda1, r1, lambda2, r2).
    lambda_g = beta_g * S_g * sum_h M[g,h] * I_h/N_h
    r_g      = gamma_g * I_g
    """
    S1, I1, S2, I2 = state
    beta1, gamma1, beta2, gamma2 = param

    lam_exp = _force_of_infection(jnp.array([I1, I2]), _N, _MIX)  # exposure per group
    lambda1 = beta1 * S1 * lam_exp[0]
    lambda2 = beta2 * S2 * lam_exp[1]
    r1 = gamma1 * I1
    r2 = gamma2 * I2
    return jnp.array([lambda1, r1, lambda2, r2])


def _stoichiometry():
    """Stoichiometry matrix S mapping reactions to state increments.
    Order of reactions: (inf_1, rec_1, inf_2, rec_2).
    """
    return jnp.array([
        [-1.0,  0.0,  0.0,  0.0],  # S1
        [ 1.0, -1.0,  0.0,  0.0],  # I1
        [ 0.0,  0.0, -1.0,  0.0],  # S2
        [ 0.0,  0.0,  1.0, -1.0],  # I2
    ])


def _project_feasible(state):
    """Project (S1,I1,S2,I2) to per-group simplex: 0<=S<=N, 0<=I<=N-S."""
    S1, I1, S2, I2 = state
    # group 1
    S1p = jnp.clip(S1, 0.0, _N[0])
    I1p = jnp.clip(I1, 0.0, _N[0] - S1p)
    # group 2
    S2p = jnp.clip(S2, 0.0, _N[1])
    I2p = jnp.clip(I2, 0.0, _N[1] - S2p)
    return jnp.array([S1p, I1p, S2p, I2p])


# -----------------------------
# Observation model
# -----------------------------

def _obs_rate(state, design):
    """
    Poisson rates for observations given current design.
    lambda_g = kappa * xi_g * rho_g * I_g / N_g
    """
    _, I1, _, I2 = state
    rate_1 = _KAPPA * sigmoid(design) * _RHO[0] * (I1 / _N[0])
    rate_2 = _KAPPA * (1 - sigmoid(design)) * _RHO[1] * (I2 / _N[1])

    rate = jnp.array([rate_1, rate_2])
    return rate + 1e-20  # shape (2,)


# -----------------------------
# Public API expected by the framework
# -----------------------------

def true_parameters():
    # Default epidemiological parameters
    beta1, gamma1 = 0.65, 0.15
    beta2, gamma2 = 0.55, 0.15
    return jnp.array([beta1, gamma1, beta2, gamma2])


def other_parameters():
    """
    Return a small additive diffusion floor Q and the integration step delta_t.
    Q is used as Q_floor in EM covariance to stabilize near-boundary states.
    """
    Q_floor = jnp.diag(jnp.array([1e-3, 1e-3, 1e-3, 1e-3]))
    R = jnp.eye(2)  # placeholder; not used for Poisson but kept for API
    delta_t = 0.1
    return Q_floor, R, delta_t


def config_param(N_parameter):
    """
    Parameter bounds for initialization (uniform):
      beta_g in [1e-3, 2.0], gamma_g in [1e-3, 1.0]
    Jittering variance scales with N_parameter.
    """
    param_bounds = jnp.array([
        [1e-1, 1.0],  # beta1
        [1e-1, 1.0],  # gamma1
        [1e-1, 1.0],  # beta2
        [1e-1, 1.0],  # gamma2
    ])
    cte_jittering = jnp.array([2, 2, 2, 2]) / (N_parameter ** 1.5)
    return param_bounds, cte_jittering


def design_bounds():
    """
    Bounds used for initialization; the actual design is projected to the simplex.
    """
    max_val = 1.0
    min_val = 0.0
    return min_val, max_val

def sample_random_design(subkey):
        # Uniform in [0, 1] for xi (design for group 1); group 2 gets 1-xi
        u = random.uniform(subkey, shape=(1,), minval=0.0, maxval=1.0)
        design = logit(u)  # logit transform to R
        return design

def fixed_design():
        # Fixed design (xi1, xi2) = (0.5, 0.5)
        u = 0.5
        design = logit(u)
        return design

def ode(state, param, design):
    """Deterministic drift f(x) = S a(x)."""
    S = _stoichiometry()
    a = _rates(state, param)
    return S @ a


def predictive_mean_cov(state, param, design, Q, delta_t):
    """State-dependent EM mean/cov with per-group projection for feasibility."""
    Q_floor, _, _ = other_parameters()
    S = _stoichiometry()
    a = _rates(state, param)

    drift = S @ a
    cov = (S * a) @ S.T  # S diag(a) S^T using broadcasting
    cov = cov * delta_t
    if Q_floor is not None:
        cov = cov + Q_floor * delta_t

    # jitter for SPD
    cov = cov + 1e-6 * jnp.eye(4)
    mean = state + delta_t * drift
    return mean, cov


def observation_mean_cov(state, R, design):
    """
    Return mean rate vector for Poisson observations and a placeholder covariance.
    """
    lam = _obs_rate(state, design)  # shape (2,)
    return lam, R  # R unused for Poisson, kept for API consistency


def init_state():
    """Start with small infections in both groups."""
    S0 = _N - _I0
    return jnp.array([S0[0], _I0[0], S0[1], _I0[1]])


def init_state_particles(subkey, N_parameter, N_state):
    x0 = init_state()
    return jnp.tile(x0, (N_parameter, N_state, 1))


def init_param_particles(subkey, N_parameter):
    param_bounds, _ = config_param(N_parameter)
    b1_min, b1_max = param_bounds[0]
    g1_min, g1_max = param_bounds[1]
    b2_min, b2_max = param_bounds[2]
    g2_min, g2_max = param_bounds[3]

    subkeys = random.split(subkey, 4)
    beta1 = random.uniform(subkeys[0], shape=(N_parameter, 1), minval=b1_min, maxval=b1_max)
    gamma1 = random.uniform(subkeys[1], shape=(N_parameter, 1), minval=g1_min, maxval=g1_max)
    beta2 = random.uniform(subkeys[2], shape=(N_parameter, 1), minval=b2_min, maxval=b2_max)
    gamma2 = random.uniform(subkeys[3], shape=(N_parameter, 1), minval=g2_min, maxval=g2_max)
    return jnp.concatenate((beta1, gamma1, beta2, gamma2), axis=1)



@jit
def sample_from_transition(subkey, previous_state, param, design, Q, delta_t):
    mean, cov = predictive_mean_cov(previous_state, param, design, Q, delta_t)
    x_new = random.multivariate_normal(subkey, mean=mean, cov=cov)
    return _project_feasible(x_new)



def sample_from_transition_complex(subkey, previous_state, param, design, Q, delta_t):
    """
    previous_state: (n_params, n_states, 4)
    param:         (n_params, 4)
    Returns:       (n_params, n_states, 4)
    """
    n_params, n_states, _ = previous_state.shape
    subkeys = random.split(subkey, n_params * n_states).reshape(n_params, n_states, -1)
    expanded_params = jnp.repeat(param[:, None, :], n_states, axis=1)

    def mean_cov(state, par):
        return predictive_mean_cov(state, par, design, Q, delta_t)

    means, covs = vmap(vmap(mean_cov, in_axes=(0, 0)), in_axes=(0, 0))(previous_state, expanded_params)
    samples = vmap(vmap(lambda k, m, C: random.multivariate_normal(k[0], mean=m, cov=C),
                        in_axes=(0, 0, 0)),
                   in_axes=(0, 0, 0))(subkeys, means, covs)
    # Project feasibility per sample
    proj = vmap(vmap(_project_feasible, in_axes=(0)), in_axes=(0))(samples)
    return proj


def eval_log_transition(current_state, previous_state, param, design, Q, delta_t):
    mean, cov = predictive_mean_cov(previous_state, param, design, Q, delta_t)
    return jscipy.stats.multivariate_normal.logpdf(current_state, mean=mean, cov=cov)


def eval_transition(current_state, previous_state, param, design, Q, delta_t):
    mean, cov = predictive_mean_cov(previous_state, param, design, Q, delta_t)
    return jscipy.stats.multivariate_normal.pdf(current_state, mean=mean, cov=cov)


def sample_observation(subkey, pred_state, R, design):
    """Poisson observation with rate lambda = kappa * xi * rho * I/N."""
    lam, _ = observation_mean_cov(pred_state, R, design)
    # Draw two independent Poisson variates
    k1, k2 = random.split(subkey)
    y1 = jnp.squeeze(random.poisson(k1, lam[0]))
    y2 = jnp.squeeze(random.poisson(k2, lam[1]))
    return jnp.array([y1, y2])



def sample_observation_complex(subkey, pred_state, R, design):
    """
    pred_state: (n_params, n_states, 4)
    Returns: (n_params, n_states, 2) counts
    """
    design = jnp.asarray(design, dtype=float)
    n_params, n_states, _ = pred_state.shape
    lam = vmap(vmap(lambda s: _obs_rate(s, design), in_axes=(0)), in_axes=(0))(pred_state, design)  # (n_params, n_states, 2)

    # Build subkeys per element
    total = n_params * n_states * 2
    keys = random.split(subkey, total).reshape(n_params, n_states, 2, -1)

    def one_pair(ks, lam2):
        y1 = random.poisson(ks[0, 0], lam2[0])
        y2 = random.poisson(ks[1, 0], lam2[1])
        return jnp.array([y1, y2], dtype=float)

    y = vmap(vmap(one_pair, in_axes=(0, 0)), in_axes=(0, 0))(keys, lam)
    return y


def eval_log_likelihood(observation, pred_state, R, design):
    """Product Poisson likelihood across the 2 groups (sum of log PMFs)."""
    lam, _ = observation_mean_cov(pred_state, R, design)
    logpmf = jscipy.stats.poisson.logpmf(observation.reshape((2,)), lam.reshape((2,)))
    return jnp.sum(logpmf)

def eval_log_likelihood_T(observation, pred_state, R, design):
    """Product Poisson likelihood across the 2 groups (sum of log PMFs)."""
    T = observation.shape[0]
    lam, _ = vmap( observation_mean_cov, in_axes=(0,None,0) )(pred_state, R, design) # (T,2)
    logpmf = vmap( jscipy.stats.poisson.logpmf, in_axes=(0,0) )(observation.reshape((T,2)), lam.reshape((T,2))) # (T,2)
    return jnp.sum(logpmf, axis=(0,1))


def eval_likelihood(observation, pred_state, R, design):
    lam, _ = observation_mean_cov(pred_state, R, design)
    logpmf = jscipy.stats.poisson.logpmf(observation.reshape((2,)), lam.reshape((2,)))
    pmf = jnp.exp(jnp.sum(logpmf))
    return pmf

def eval_likelihood_T(observation, pred_state, R, design):
    T = observation.shape[0]
    lam, _ = vmap( observation_mean_cov, in_axes=(0,None,0) )(pred_state, R, design) # (T,2)
    logpmf = vmap( jscipy.stats.poisson.logpmf, in_axes=(0,0) )(observation.reshape((T,2)), lam.reshape((T,2))) # (T,2)
    pmf = jnp.exp(jnp.sum(logpmf, axis=(0,1)))
    return pmf  
