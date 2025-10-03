import jax.numpy as jnp
import jax.scipy as jscipy
from jax import random, vmap
from jax import jit

# ============================================================
# Moving source location model (state = [px, py, phi]; params = [vx, vy])
# Known constant: v_phi (angular rate). Design = sensor orientations (angles), affects observation only
# ============================================================

# -----------------------------
# Sensor array & observation hyperparameters (defaults)
# -----------------------------
_SENSORS = jnp.array([
    [3.0, 0.0],
    [0.0, 3.0]
])  # shape (J,2) - location of sensors in 2D
_ALPHA = jnp.full(_SENSORS.shape[0], 5.0)  # per-sensor strengths, shape (J,)
_B = 0.1                               # background level > 0
_M_SAT = 0.1                           # saturation constant m > 0
_D_DIR = 1.0                           # cardioid parameter d in [0,1)
_SIGMA_Y = 0.1                         # observation noise std (log domain)
_K_EXPONENT = 4.0                     # directivity exponent > 0
_DESIGN_DIM = _SENSORS.shape[0]  # design dimension (number of sensors)
_STATE_DIM = 3    # state dimension (px, py, phi)
_PARAM_DIM = 2    # parameter dimension (vx, vy)
_OBS_DIM = _SENSORS.shape[0]  # observation dimension (J sensors

# Known angular rate (heading change per unit time); can be overridden via set_known_vphi()
_VPHI = 0.3

# Current design (sensor orientations); must be set before calling observation functions
_CURRENT_DESIGN = None  # shape (J,)

def set_sensor_array(sensors, alpha=None, b=None, m=None, d=None, sigma_y=None):
    """
    Configure sensor geometry and observation hyperparameters.
    sensors : array (J,2)
    alpha   : array (J,), per-sensor strengths
    b       : float > 0, background
    m       : float > 0, saturation constant
    d       : float in [0,1), cardioid directivity parameter
    sigma_y : float > 0, std of log-observation noise
    """
    global _SENSORS, _ALPHA, _B, _M_SAT, _D_DIR, _SIGMA_Y
    _SENSORS = jnp.array(sensors)
    if alpha is not None:
        _ALPHA = jnp.array(alpha)
    if b is not None:
        _B = float(b)
    if m is not None:
        _M_SAT = float(m)
    if d is not None:
        _D_DIR = float(d)
    if sigma_y is not None:
        _SIGMA_Y = float(sigma_y)

def set_current_design(design):
    """Wrap angles to [-pi, pi] and cache the feasible design.

    Returns
    -------
    jnp.ndarray
        Sensor orientation vector after enforcing angular bounds.
    """

    # global _CURRENT_DESIGN

    design = jnp.asarray(design, dtype=float)
    wrapped = _wrap_pi(design)
    min_val, max_val = design_bounds()
    projected = jnp.clip(wrapped, min_val, max_val)

    # _CURRENT_DESIGN = projected
    return projected

# -----------------------------
# Helpers
# -----------------------------

def _wrap_pi(theta):
    return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

def _bearing(p, s):
    # Bearing from sensor s to source position p
    return jnp.arctan2(p[1] - s[1], p[0] - s[0])  # in (-pi, pi]

# def _directivity(delta):
#     # Cardioid: D(delta) = (1 + d cos delta)/(1 + d) > 0
#     return (1.0 + _D_DIR * jnp.cos(delta)) / (1.0 + _D_DIR)

def _intensity_mu(p, design):
    """
    Compute mean intensities mu_j (not in log) for all sensors, given source position p
    and sensor orientations 'design' (shape (J,)).
    mu_j = b + alpha_j / (m + ||p - s_j||^2) * D( delta_j )
    """
    # Vectorize over sensors
    def per_sensor(s, a, xi):
        r2 = jnp.sum((p - s) ** 2)
        psi = _bearing(p, s)
        #directivity = jnp.power((1.0 + (_D_DIR * jnp.cos(xi - psi))) / (1.0 + _D_DIR), _K_EXPONENT)
        directivity = ((1.0 + (_D_DIR * jnp.cos(xi - psi))) / (1.0 + _D_DIR)) ** _K_EXPONENT
        return _B + a / (_M_SAT + r2 + 1e-12) * directivity

    return vmap(per_sensor)(_SENSORS, _ALPHA, design)  # shape (J,)

# -----------------------------
# Public API expected by the framework
# -----------------------------

def true_parameters():
    # Motion parameters to learn: vx, vy (phi evolves with known v_phi)
    vx, vy = 1.0, 1.0
    return jnp.array([vx, vy])

def other_parameters():
    # Process noise covariance Q on [px, py, phi] and observation covariance R on log-intensities
    Q = jnp.diag(jnp.array([2e-1, 2e-1, 1e-2]))
    J = _SENSORS.shape[0]
    R = jnp.eye(J) * (_SIGMA_Y ** 2)
    delta_t = 0.1
    return Q, R, delta_t

def config_param(N_parameter):
    # Parameter bounds for initialization: vx, vy
    param_bounds = jnp.array([[0.5, 1.5], [0.5, 1.5]])
    # Jittering variance per parameter (scaled with N_parameter)
    cte_jittering = jnp.array([0.15, 0.15]) / (N_parameter ** 1.5)
    return param_bounds, cte_jittering

def design_bounds():
    # Sensor orientations in [-pi, pi]
    max_val = jnp.pi
    min_val = -jnp.pi
    return min_val, max_val

def sample_random_design(subkey):
       # Random design: uniform in [-pi, pi] for J sensors
       J = _SENSORS.shape[0]
       _, subkey = random.split(subkey)
       min_val, max_val = design_bounds()
       return jnp.asarray(random.uniform(subkey, shape=(J,), minval=min_val, maxval=max_val))

def fixed_design():
        # Fixed design (angles) for J sensors at orientation 0
        J = _SENSORS.shape[0]
        return jnp.zeros(J, dtype=float)

def ode(state, param, design):
    """
    Constant-turn-rate model with known angular rate v_phi:
    x_t = x_{t-1} + dt * [vx cos(phi), vy sin(phi), v_phi]
    (Design does not affect transition; kept for API compatibility.)
    """
    px, py, phi = state
    vx, vy = param[0], param[1]
    return jnp.array([vx * jnp.cos(phi), vy * jnp.sin(phi), _VPHI])

def predictive_mean_cov(state, param, design, Q, delta_t):
    # state in R^3 = [px, py, phi]; params = [vx, vy]
    mean = state + delta_t * ode(state, param, design)
    # wrap heading to (-pi, pi]
    mean = mean.at[2].set(_wrap_pi(mean[2]))
    cov = Q * delta_t
    return mean, cov

def observation_mean_cov(state, R, design):
    """
    Return mean and covariance of log-observation.
    state: array-like of shape (3,) giving [px, py, phi].
    design: array of shape (J,), sensor orientations in radians.
    """
    p = state[:2]
    mu = _intensity_mu(p, design)  # shape (J,)
    mean = jnp.log(mu)             # log-intensity
    cov = R
    return mean, cov

def init_state():
    # [px, py, phi] near origin with heading ~ 0
    return jnp.array([0.0, 0.0, 0.0])

def init_state_particles(subkey, N_parameter, N_state):
    # particles in R^3: [px, py, phi]
    subkeys = random.split(subkey, 2)
    init_state_arr = jnp.broadcast_to(init_state(), (N_parameter, N_state, 3))
    noise = 0.01 * random.normal(subkeys[0], shape=(N_parameter, N_state, 3))
    return init_state_arr + noise

def init_param_particles(subkey, N_parameter):
    param_bounds, _ = config_param(N_parameter)
    (vx_min, vx_max), (vy_min, vy_max) = param_bounds
    subkeys = random.split(subkey, 2)
    vx_particles = random.uniform(subkeys[0], shape=(N_parameter, 1), minval=vx_min, maxval=vx_max)
    vy_particles = random.uniform(subkeys[1], shape=(N_parameter, 1), minval=vy_min, maxval=vy_max)
    return jnp.concatenate((vx_particles, vy_particles), axis=1)

def init_design(subkey):
    """
    Initialize a design schedule: orientations for all J sensors at each time step.
    Returns an array of shape (T, J).
    """
    min_val, max_val = design_bounds()
    J = _SENSORS.shape[0]
    return random.uniform(subkey, shape=(J,), minval=min_val, maxval=max_val)


@jit
def sample_from_transition(subkey, previous_state, param, design, Q, delta_t):
    mean, cov = predictive_mean_cov(previous_state, param, design, Q, delta_t)
    _, subkey = random.split(subkey)
    return random.multivariate_normal(subkey, mean=mean, cov=cov)

def sample_from_transition_complex(subkey, previous_state, param, design, Q, delta_t):
    """
    previous_state: (n_params, n_states, 3)
    param: (n_params, 2) with known v_phi used internally
    design: kept for API; dynamics do not depend on it
    """
    n_params, n_states, _ = previous_state.shape
    subkeys = random.split(subkey, n_params * n_states).reshape(n_params, n_states, -1)
    expanded_params = jnp.repeat(param[:, None, :], n_states, axis=1)

    def mean_cov(state, par):
        return predictive_mean_cov(state, par, design, Q, delta_t)

    means, covs = vmap(vmap(mean_cov, in_axes=(0, 0)), in_axes=(0, 0))(previous_state, expanded_params)
    return vmap(vmap(random.multivariate_normal, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(subkeys, means, covs)

def eval_log_transition(current_state, previous_state, param, design, Q, delta_t):
    mean, cov = predictive_mean_cov(previous_state, param, design, Q, delta_t)
    return jscipy.stats.multivariate_normal.logpdf(current_state, mean=mean, cov=cov)

def eval_transition(current_state, previous_state, param, design, Q, delta_t):
    mean, cov = predictive_mean_cov(previous_state, param, design, Q, delta_t)
    return jscipy.stats.multivariate_normal.pdf(current_state, mean=mean, cov=cov)

def sample_observation(subkey, pred_state, R, design):
    mean, cov = observation_mean_cov(pred_state, R, design)
    _, subkey = random.split(subkey)
    return random.multivariate_normal(subkey, mean=mean, cov=cov)

def sample_observation_complex(subkey, pred_state, R, design):
    """
    pred_state: (n_params, n_states, 3)
    Returns: (n_params, n_states, J) draws of log-intensity
    """
    n_params, n_states, _ = pred_state.shape
    subkeys = random.split(subkey, n_params * n_states).reshape(n_params, n_states, -1)

    def mean_cov(state):
        return observation_mean_cov(state, R, design)

    means, covs = vmap(vmap(mean_cov, in_axes=(0)), in_axes=(0))(pred_state)
    return vmap(vmap(random.multivariate_normal, in_axes=(0, 0, None)),
                in_axes=(0, 0, None))(subkeys, means, covs)

def eval_log_likelihood(observation, pred_state, R, design):
    mean, cov = observation_mean_cov(pred_state, R,design)
    return jscipy.stats.multivariate_normal.logpdf(observation, mean=mean, cov=cov)

def eval_log_likelihood_T(observation, pred_state, R, design):
    """Product Gaussian likelihood across T time steps (sum of log PDFs)."""
    T = observation.shape[0]
    mean, cov = vmap( observation_mean_cov, in_axes=(0,None,0) )(pred_state, R, design) # (T,J)
    return jnp.sum(vmap( jscipy.stats.multivariate_normal.logpdf, in_axes=(0,0,None) )(observation, mean, cov))

def eval_likelihood(observation, pred_state, R, design):
    mean, cov = observation_mean_cov(pred_state, R, design)
    return jscipy.stats.multivariate_normal.pdf(observation, mean=mean, cov=cov)

def eval_likelihood_T(observation, pred_state, R, design):
    T = observation.shape[0]
    mean, cov = vmap( observation_mean_cov, in_axes=(0,None,0) )(pred_state, R, design) # (T,J)
    return jnp.prod(vmap( jscipy.stats.multivariate_normal.pdf, in_axes=(0,0,None) )(observation, mean, cov))
