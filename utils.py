"""
Some random functions that are useful for the project.
"""
import jax.numpy as jnp
from jax import random, jit, lax, debug

def sanitize_logs(x, floor=-1e12):
    return jnp.where(jnp.isfinite(x), x, floor)         # replace NaN/Inf

def sanitize_gradients(x, replace=0.0):
    """Replace NaN/Inf with a finite value element-wise."""
    return jnp.where(jnp.isfinite(x), x, jnp.array(replace, x.dtype))

def truncated_gaussian_sample(subkey, mean, variance, lower_bound, upper_bound):
    std_dev = jnp.sqrt(variance)
    a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev  # Calculate the a and b parameters for truncnorm
    _, subkey = random.split(subkey)
    sample_standard = random.truncated_normal(key=subkey,lower=a,upper=b)
    return sample_standard * std_dev + mean


def catching_errors_weights(log_likelihood):
    """
    Catch potential problems in the weights. If there are NaN or Inf values 
    in the log-likelihood, we set them to a very low value (-1e6).

    Parameters:
    log_likelihood (array): Log-likelihood of the observation given a sample. Size: (n_particles,)
   
    Returns:
    tuple: log-likelihood of the observation after removing errors. Size: (n_particles,)
    """
    # Handle NaN or Inf values in log-likelihood
    is_bad =  ~jnp.isfinite(log_likelihood)
    log_likelihood = jnp.where(is_bad, -1e6, log_likelihood)

    return log_likelihood


def mse_temporal_sequence(ground_truth, estimate,type='mse'):
    
    se = jnp.sum((estimate - ground_truth)**2, axis=1) # squared error
    norm_ground_truth = jnp.sum((ground_truth)**2, axis=1)
    nse = se / norm_ground_truth # normalized squared error

    if type == 'nmse':
        value_e = nse
    elif type == 'mse':
        value_e = se
    elif type == 'rmse':
        value_e = jnp.sqrt(se)
    elif type == 'rnmse':
        value_e = jnp.sqrt(nse)
    else:
        raise ValueError("Invalid type. Choose 'nmse', 'mse', 'rmse' or 'rnmse'.")

    return value_e


def ess(w, type='chi2'):
    """
    Compute the effective sample size (ESS) based on the weights.

    Parameters:
    w (array): Weights. Size: (N,).
    type (str): Type of ESS computation. Options: 'chi2' or 'kl'.

    Returns:
    float: Effective sample size.
    """
    N = w.shape[0]
    if type == 'chi2':
        ess = 1 / jnp.sum(w**2)
    elif type == 'kl': # recommended when infinite variance
        ess = N / jnp.exp(jnp.sum(w * jnp.log(N*w))/N)
    else: 
        raise ValueError("Invalid type. Choose 'chi2' or 'kl'.")
    return ess


def regularize_covariance(cov_matrix, epsilon=1e-6):
    """
    Regularize a covariance matrix by adding a small value to the diagonal.
    
    Args:
        cov_matrix: The covariance matrix to regularize
        epsilon: The value to add to the diagonal
        
    Returns:
        Regularized positive semidefinite covariance matrix
    """
    # Check if matrix needs regularization
    eigenvalues = jnp.linalg.eigvalsh(cov_matrix)
    needs_regularization = jnp.min(eigenvalues) < epsilon
    
    # Add epsilon to diagonal if needed
    return jnp.where(
        needs_regularization,
        cov_matrix + jnp.eye(cov_matrix.shape[0]) * epsilon,
        cov_matrix
    )

