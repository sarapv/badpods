import time
import datetime
import argparse
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import random, lax, block_until_ready
from jax import jit, grad, vmap
from jax.scipy.special import logsumexp
from utils import truncated_gaussian_sample, ess, sanitize_logs, sanitize_gradients
from jax import debug as jax_debug

from jax.scipy.special import logit
from jax.nn import sigmoid


MODEL_TYPE = os.environ.get('MODEL_TYPE', 'sir').lower()
if MODEL_TYPE == 'sir':
    # import group_SIR as model
    import group_SIR_2param as model
elif MODEL_TYPE == 'source':
    import source as model
else:
    import group_SIR_2param as model
    MODEL_TYPE = 'sir'



_grad_eval_likelihood = grad(model.eval_likelihood, argnums=3)
_grad_log_eval_likelihood = grad(model.eval_log_likelihood, argnums=3)


@partial(jit, static_argnums=())
def _batched_log_likelihood(design, R, obs, states):
    def eval_obs(obs_row):
        return vmap(lambda state: model.eval_log_likelihood(obs_row, state, R, design))(states)

    return vmap(eval_obs)(obs)


@partial(jit, static_argnums=())
def _batched_grad_likelihood(design, R, obs, states):
    def eval_obs(obs_row):
        return vmap(lambda state: _grad_eval_likelihood(obs_row, state, R, design))(states)

    return vmap(eval_obs)(obs)


@partial(jit, static_argnums=())
def _batched_grad_log_likelihood(design, R, obs, states):
    return vmap(lambda obs_row, state_row: _grad_log_eval_likelihood(obs_row, state_row, R, design))(obs, states)



def NPF_state_layer(
                    subkey, 
                    state_particles, 
                    observation, 
                    parameters, 
                    design, 
                    delta_t
                    ):
     
    
    N_state = state_particles.shape[0]          # Number of particles (state layer)
    Q, R, _ = model.other_parameters()   
    
    # Prediction step
    subkeys = random.split(subkey,N_state+1)
    _, subkey = random.split(subkeys[-1])
    state_ps_pred = vmap(
                            model.sample_from_transition,
                            in_axes=(0,0,None,None,None,None)
                        )(
                            subkeys[:-1],
                            state_particles, 
                            parameters, 
                            design, 
                            Q,
                            delta_t
                        )

    # Compute the log-likelihood
    logws_state = sanitize_logs(
                                vmap(
                                    model.eval_log_likelihood, 
                                    in_axes=(None,0,None,None)
                                    )(
                                    observation, 
                                    state_ps_pred, 
                                    R,
                                    design
                                    )
                                )

    # Compute weights
    norm_ws = jnp.exp(logws_state - logsumexp(logws_state))

    # Compute log outer weight (for the parameters)
    logws_param = sanitize_logs(
                                logsumexp(logws_state) 
                                - jnp.log(N_state)
                                )

    # Compute state estimate
    state_est = jnp.dot(state_ps_pred.T, norm_ws)

    # Resampling state
    _, subkey = random.split(subkey)
    idx_state = random.choice(subkey, N_state, shape=(N_state,), p=norm_ws)
    state_ps_res = state_ps_pred[idx_state]

    return state_ps_pred, state_ps_res, state_est, logws_param

def NPF_param_layer(
        subkey,
        true_parameters,
        delta_t,
        T_run, 
        N_state, 
        N_param, 
        pbar=None,
        use_wandb=False,
        save_results=False,
        verbose=False,
        results_dir=None,
        random_design=False,
        fixed_design=False,
        K_opt=50,
):
    model_type = MODEL_TYPE
    # Covariance matrices
    Q, R, _ = model.other_parameters()

    # Dimension of variables
    param_dim = true_parameters.shape[0]  
    state_dim = Q.shape[0]
    obs_dim = R.shape[0]
        
    # Split random keys for algorithm and ground truth
    algo_key, gt_key = random.split(subkey)

    # Initialize particles for theta, x
    param_key, state_key, algo_key = random.split(algo_key,3)
    param_ps = model.init_param_particles(param_key, N_param)
    state_ps_nested = model.init_state_particles(state_key, N_param, N_state)

    # Initialize design (random)
    design_key, algo_key = random.split(algo_key, 2)
    chosen_design = model.sample_random_design(design_key)
    

    # Initialize x_0, y_0 (ground truth)
    prev_true_state = model.init_state()
    # Initialize hat{x}_0, hat{theta}_0 (estimates)
    prev_param_est = model.init_param_particles(param_key, 1).squeeze()
    prev_state_est = model.init_state()
    
    # ESS, EIG
    param_ess_t = N_param
    eig_est_t = 0.0
    total_eig_t = 0.0
   
    # Histories
    true_states_T_history = jnp.zeros((T_run+1, state_dim)) 
    true_states_T_history = true_states_T_history.at[0].set(model.init_state()) 
    param_est_T_history = jnp.zeros((T_run+1,param_dim))  
    param_est_T_history = param_est_T_history.at[0].set(prev_param_est) 
    state_est_T_history = jnp.zeros((T_run+1, state_dim))
    state_est_T_history = state_est_T_history.at[0].set(prev_state_est)


    # Wandb initialization OR results saving
    if use_wandb:
        wandb_eig_rows = []
        wandb_eig_columns = None
        wandb_design1_rows = []
        wandb_design1_columns = None
        wandb_design2_rows = []
        wandb_design2_columns = None

    if save_results:
        designs_T_history = jnp.zeros((T_run, initial_design.shape[0]))
        designs_T_history = designs_T_history.at[0].set(initial_design)          
        observations_T_history = jnp.zeros((T_run, obs_dim))
        param_ess_T_history = jnp.zeros((T_run+1,))  
        param_ess_T_history = param_ess_T_history.at[0].set(param_ess_t) 
        eig_est_T_history = jnp.zeros((T_run+1,))  
        eig_est_T_history = eig_est_T_history.at[0].set(eig_est_t) 
        total_eig_T_history = jnp.zeros((T_run+1,))  
        total_eig_T_history = total_eig_T_history.at[0].set(total_eig_t)

    


    # Warm-up for design optimization
    if not random_design and not fixed_design:
        K_optimization = 3
        warm_key, algo_key = random.split(algo_key)
        _, _, warm_designs_hist, warm_eig_hist = _run_design_optimization(
            subkey=warm_key,
            init_design=chosen_design,
            delta_t=delta_t,
            state_ps_nested=state_ps_nested,
            param_ps=param_ps,
            K=K_optimization
        )
        block_until_ready(warm_eig_hist)




    # Vectorized function to call the state layer
    vmap_NPF_state_layer_step = vmap(
                                    NPF_state_layer, 
                                    in_axes=(0,0,None,0,None,None)
                                    )

    # Time loop, stopping at each new observation
    for t in range(T_run):

        # RANDOM DESIGN
        if random_design:
            design_key, eig_key, algo_key = random.split(algo_key, 3)

            chosen_design = model.sample_random_design(design_key)
            

            _, estimator_eig = gradient_EIG(
                subkey=eig_key,
                param_ps=param_ps,
                state_ps_nested=state_ps_nested,
                design=chosen_design,
                delta_t=delta_t,
                gradient_estimation=False,
            )
            eig_est_t = estimator_eig
            total_eig_t += eig_est_t


        # FIXED DESIGN
        elif fixed_design:
            eig_key, algo_key = random.split(algo_key, 2)

            chosen_design = model.fixed_design()
            

            _, estimator_eig = gradient_EIG(
                subkey=eig_key,
                param_ps=param_ps,
                state_ps_nested=state_ps_nested,
                design=chosen_design,
                delta_t=delta_t,
                gradient_estimation=False,
            )
            eig_est_t = estimator_eig
            total_eig_t += eig_est_t


        # OPTIMIZED DESIGN (BED)
        else:


             # Warm-up design optimization
            
            if model_type == 'sir':
                K_warmup = 1
                N_search_designs = 20
            elif model_type == 'source':
                K_warmup = 5
                N_search_designs = 20

            warmup_designs = jnp.zeros((N_search_designs, model._DESIGN_DIM))
            warmup_eig = jnp.zeros(N_search_designs)

            warm_key, algo_key = random.split(algo_key, 2)
            designs_keys = random.split(algo_key, N_search_designs+2)
            algo_key = designs_keys[-1]

            lnspc = jnp.linspace(-6, 6, N_search_designs)

            for i in range(N_search_designs):
                warmup_design = model.sample_random_design(designs_keys[i])
                # warmup_design = float(lnspc[i])
                chosen_design, best_eig, _, _ = _run_design_optimization(
                    subkey=warm_key,
                    init_design=warmup_design,
                    delta_t=delta_t,
                    state_ps_nested=state_ps_nested,
                    param_ps=param_ps,
                    K=K_warmup
                )
                warmup_eig = warmup_eig.at[i].set(best_eig)
                warmup_designs = warmup_designs.at[i].set(chosen_design)
                # print(f"Warm-up design {i+1}/{N_search_designs} {sigmoid(warmup_design)}: final design: {sigmoid(chosen_design)} - EIG: {best_eig:.4f}")
            # print(f"Warm-up design optimization done. Starting main optimization...\n")
            algo_key, subkey = random.split(algo_key)
            initial_design = warmup_designs[jnp.argmax(warmup_eig)]

    

            # Optimization
            chosen_design, best_eig, intermediate_designs, intermediate_eig = _run_design_optimization(
                subkey=algo_key,
                init_design=initial_design,
                delta_t=delta_t,
                state_ps_nested=state_ps_nested,
                param_ps=param_ps,
                K=K_opt
            )

            if best_eig > 0:
                eig_est_t = best_eig
            else:
                eig_est_t = 1e-12
            total_eig_t += eig_est_t

            if use_wandb:
                intermediate_eig_np = np.asarray(intermediate_eig)
                if wandb_eig_columns is None:
                    wandb_eig_columns = ["time_step"] + [f"iter_{k}" for k in range(intermediate_eig_np.shape[0])]
                wandb_eig_rows.append([t] + [float(val) for val in intermediate_eig_np])

                design_history = np.asarray(intermediate_designs)
                if design_history.ndim == 2 and design_history.shape[1] >= 2:
                    if wandb_design1_columns is None:
                        wandb_design1_columns = ["time_step"] + [f"iter_{k}" for k in range(intermediate_eig_np.shape[0])]
                    wandb_design1_rows.append([t] + [float(val) for val in design_history[:, 0]])

                    if wandb_design2_columns is None:
                        wandb_design2_columns = ["time_step"] + [f"iter_{k}" for k in range(intermediate_eig_np.shape[0])]
                    wandb_design2_rows.append([t] + [float(val) for val in design_history[:, 1]])

        print(f"Time {t+1}/{T_run} - Chosen design: {sigmoid(chosen_design)} - EIG: {eig_est_t:.4f} - Total EIG: {total_eig_t:.4f}\n")
                
        # Generate ground truth
        gt_state_key, gt_obs_key = random.split(random.fold_in(gt_key, t), 2)
        new_true_state = model.sample_from_transition(
                                            gt_state_key, 
                                            prev_true_state, 
                                            true_parameters, 
                                            chosen_design, 
                                            Q, 
                                            delta_t
                                            )
        new_observation= model.sample_observation(
                                            gt_obs_key, 
                                            new_true_state, 
                                            R,
                                            chosen_design
                                            )

            
        
        # Jitter parameter particles
        jittering_key, algo_key = random.split(algo_key,2)
        param_ps = jittering_parameter_particles(
                                                jittering_key, 
                                                param_ps
                                                )
        

        
        # Run the state layer  
        state_layer_subkeys = random.split(algo_key,N_param+1)
        _, algo_key = random.split(state_layer_subkeys[-1])
        _, state_ps_second_layer, state_est_second_layer, logws_param = \
            vmap_NPF_state_layer_step(
                                        state_layer_subkeys[0:-1], 
                                        state_ps_nested, 
                                        new_observation, 
                                        param_ps,
                                        chosen_design,
                                        delta_t
                                    )
      
        # Compute weights for the parameter particles
        n_ws_param = jnp.exp(logws_param - logsumexp(logws_param))

        # ESS parameter layer
        param_ess_t = ess(n_ws_param,'chi2')
        
        # Estimates at time t
        param_est_t = jnp.dot(param_ps.T, n_ws_param)         
        state_est_t = jnp.dot(state_est_second_layer.T,n_ws_param)
           

        # Resampling everything
        resampling_key, algo_key = random.split(algo_key,2)
        idx_param = random.choice(
                                    resampling_key, 
                                    N_param, 
                                    shape=(N_param,), 
                                    p=n_ws_param
                                )
        param_ps = param_ps[idx_param]
        state_ps_nested = state_ps_second_layer[idx_param]

        # --- Per-step RMSE (logged each time step) ---
        # RMSE for state estimate at time t (vs. true state at time t)
        rmse_state_step = float(jnp.sqrt(jnp.mean((state_est_t - new_true_state)**2)))
        # RMSE for parameter estimate at time t (vs. true static parameters)
        rmse_param_step = float(jnp.sqrt(jnp.mean((param_est_t - true_parameters)**2)))

        if use_wandb:
            model_type = MODEL_TYPE
            if model_type == 'sir':
                wandb.log({
                    # Parameter estimates
                    "beta1_estimates": param_est_t[0],
                    "gamma1_estimates": param_est_t[1],
                    "beta2_estimates": param_est_t[2],
                    "gamma2_estimates": param_est_t[3],
                    # State estimates
                    "S1_estimates": state_est_t[0],
                    "I1_estimates": state_est_t[1],
                    "S2_estimates": state_est_t[2],
                    "I2_estimates": state_est_t[3],
                    # Parameter particles (last time step)
                    "beta1_particles": param_ps[:,0],
                    "gamma1_particles": param_ps[:,1],
                    "beta2_particles": param_ps[:,2],
                    "gamma2_particles": param_ps[:,3],
                    # True values of parameters and states
                    "beta1_true": true_parameters[0],
                    "gamma1_true": true_parameters[1],
                    "beta2_true": true_parameters[2],
                    "gamma2_true": true_parameters[3],
                    "S1_true": new_true_state[0],
                    "I1_true": new_true_state[1],
                    "S2_true": new_true_state[2],
                    "I2_true": new_true_state[3],
                    # Other metrics
                    "parameter_ess": param_ess_t,
                    "rmse_state_step": rmse_state_step,
                    "rmse_parameters_step": rmse_param_step,
                    "total_eig": total_eig_t,
                    # Chosen designs at time t
                    "design_1": sigmoid(chosen_design),
                    "design_2": 1 - sigmoid(chosen_design),
                })
            elif model_type == 'source':
                wandb.log({
                    # Parameter estimates
                    "vx_estimates": param_est_t[0],
                    "vy_estimates": param_est_t[1],
                    # State estimates
                    "p1_estimates": state_est_t[0],
                    "p2_estimates": state_est_t[1],
                    "phi_estimates": state_est_t[2],
                    # Parameter particles (last time step)
                    "vx_particles": param_ps[:,0],
                    "vy_particles": param_ps[:,1],
                    # True values of parameters and states
                    "vx_true": true_parameters[0],
                    "vy_true": true_parameters[1],
                    "p1_true": new_true_state[0],
                    "p2_true": new_true_state[1],
                    "phi_true": new_true_state[2],
                    # Other metrics
                    "parameter_ess": param_ess_t,
                    "rmse_state_step": rmse_state_step,
                    "rmse_parameters_step": rmse_param_step,
                    "total_eig": total_eig_t,
                    # Chosen designs at time t
                    "design_1": chosen_design[0],
                    "design_2": chosen_design[1],
                    # bearing at time t (true bearing from each sensor to source)
                    "bearing_1": model._wrap_pi(chosen_design[0] - model._bearing(new_true_state[0:2], model._SENSORS[0])),
                    "bearing_2": model._wrap_pi(chosen_design[1] - model._bearing(new_true_state[0:2], model._SENSORS[1])),
                })
        
        if save_results:
            designs_T_history = designs_T_history.at[t].set(chosen_design)
            eig_est_T_history = eig_est_T_history.at[t].set(eig_est_t)
            total_eig_T_history = total_eig_T_history.at[t].set(eig_est_t if t == 0 else total_eig_T_history[t - 1] + eig_est_t)
            observations_T_history = observations_T_history.at[t+1].set(new_observation)
            param_ess_T_history = param_ess_T_history.at[t].set(param_ess_t) 
                

        
        true_states_T_history = true_states_T_history.at[t+1].set(new_true_state) 
        param_est_T_history = param_est_T_history.at[t].set(param_est_t)  
        state_est_T_history = state_est_T_history.at[t].set(state_est_t)

        prev_true_state = new_true_state
        prev_param_est = param_est_t
        prev_state_est = state_est_t


        # Print progress
        if pbar is not None:
            pbar.update(1)


    # Stop printing progress
    if pbar is not None:
        pbar.close()

    if use_wandb and wandb_eig_rows and model._DESIGN_DIM >= 1:
        log_payload = {"optimization_eig": wandb.Table(columns=wandb_eig_columns, data=wandb_eig_rows)}

        if wandb_design1_rows:
            log_payload["optimization_design_1"] = wandb.Table(columns=wandb_design1_columns, data=wandb_design1_rows)

        if wandb_design2_rows:
            log_payload["optimization_design_2"] = wandb.Table(columns=wandb_design2_columns, data=wandb_design2_rows)

        wandb.log(log_payload)
    
    if save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"results_{MODEL_TYPE}_{timestamp}.npz")
        np.savez(
            filename,
            designs_T_history=np.asarray(designs_T_history),
            true_states_T_history=np.asarray(true_states_T_history),
            observations_T_history=np.asarray(observations_T_history),
            param_est_T_history=np.asarray(param_est_T_history),
            state_est_T_history=np.asarray(state_est_T_history),
            param_ess_T_history=np.asarray(param_ess_T_history),
            eig_est_T_history=np.asarray(eig_est_T_history),
            total_eig_T_history=np.asarray(total_eig_T_history)
        )
        if verbose:
            print(f"Results saved to {filename}")



    return param_est_T_history, state_est_T_history, chosen_design, true_states_T_history



def jittering_parameter_particles(
        subkey, 
        param_ps
    ):
   
    # Number of particles
    N_param = param_ps.shape[0]
    # Parameter bounds and constant for jittering
    param_bounds, cte_jittering = model.config_param(N_param) # Parameter bounds (uniform pdf for initialization)

    # Extract parameter bounds
    param_lower_bound = param_bounds[:,0]
    param_upper_bound = param_bounds[:,1]

    # Vectorize truncated Gaussian sampling
    vectorized_truncated_gaussian_sample = vmap(
                                                truncated_gaussian_sample, 
                                                in_axes=(0,0,None,None,None)
                                                )

     # Jitter the parameter particles
    subkeys = random.split(subkey,N_param)
    param_ps_jittered = vectorized_truncated_gaussian_sample(
                                                            subkeys, 
                                                            param_ps, 
                                                            cte_jittering,
                                                            param_lower_bound, 
                                                            param_upper_bound
                                                            )
   
    return param_ps_jittered


def samples_outerExpectation_v2(
                            subkey, 
                            param_ps, 
                            state_ps_nested, 
                            design, 
                            delta_t
                            ):

    Q, R, _ = model.other_parameters()
    state_key, obs_key = random.split(subkey, 2)

    # Sampling states
    state_ps_outerExp = model.sample_from_transition_complex(
                                                        state_key,
                                                        state_ps_nested,
                                                        param_ps,
                                                        design,
                                                        Q,
                                                        delta_t
                                                        )

    # Sampling observations
    obs_ps_outerExp = model.sample_observation_complex(
                                                obs_key,
                                                state_ps_outerExp,
                                                R,
                                                design
                                                )

    return state_ps_outerExp, obs_ps_outerExp
        

#@jit
def samples_Expectation(
                            subkey, 
                            param_ps, 
                            state_ps_nested, 
                            design, 
                            delta_t
                            ):
    # Number of samples 
    Q, R, _ = model.other_parameters()
    N_param = param_ps.shape[0]
    N_state = state_ps_nested.shape[1]

    # Vmaps
    vmap_sampling_state = vmap(
                                vmap(
                                    model.sample_from_transition, 
                                    in_axes=(0,0,None,None,None,None)
                                    ),
                                in_axes=(0,0,0,None,None,None)
                                )
    vmap_sampling_observation = vmap(
                                    vmap(
                                        model.sample_observation, 
                                        in_axes=(0,0,None,None)
                                        ),
                                    in_axes=(0,0,None,None)
                                    )

    # Sampling states
    state_keys = random.split(subkey,N_param*(N_state+1)).reshape(N_param, N_state+1, 2)
    _, subkey = random.split(state_keys[-1,-1])
    state_ps_outerExp = vmap_sampling_state(
                                            state_keys[:,:-1],
                                            state_ps_nested,
                                            param_ps,
                                            design,
                                            Q,
                                            delta_t
                                            )

    # Sampling observations
    obs_keys = random.split(subkey, N_param*(N_state+1)).reshape(N_param, N_state+1, 2)
    _, subkey = random.split(obs_keys[-1,-1])
    obs_ps_outerExp = vmap_sampling_observation(
                                                obs_keys[:,:-1],
                                                state_ps_outerExp,
                                                R,
                                                design
                                                )
    
     # Jittering parameters
    jittering_key, subkey = random.split(subkey,2)
    param_ps_innerExp_joint = jittering_parameter_particles(
                                                            jittering_key, 
                                                            param_ps
                                                            )
                            
    # Sampling states (joint distribution)
    # state_keys = random.split(subkey, N_param*(N_state+1)).reshape(N_param, N_state+1, 2)
    # _, subkey = random.split(state_keys[-1,-1])
    state_ps_innerExp_joint = vmap_sampling_state(
                                                state_keys[:,:-1],
                                                state_ps_nested,
                                                param_ps_innerExp_joint,
                                                design,
                                                Q,
                                                delta_t
                                                )

    return state_ps_outerExp, obs_ps_outerExp, state_ps_innerExp_joint, param_ps_innerExp_joint




def gradient_EIG(
        subkey,
        param_ps,
        state_ps_nested,
        design,
        delta_t,
        gradient_estimation=False
    ):
    """Estimate the expected information gain (EIG) and, optionally, its gradient.

    Parameters
    ----------
    subkey : jax.random.PRNGKey
        Random seed used to generate the Monte-Carlo samples.
    param_ps : jnp.ndarray (N_param, param_dim)
        Parameter particles at time ``t``.
    state_ps_nested : jnp.ndarray (N_param, N_state, state_dim)
        State particles nested inside each parameter particle.
    design : jnp.ndarray (...)
        Current design variable to be optimised.
    delta_t : float
        Discretisation step of the state transition.
    gradient_estimation : bool, default False
        When True, also compute the gradient of the EIG w.r.t. the design.

    Returns
    -------
    estimator_gradient_eig : jnp.ndarray | None
        Scalar gradient estimate (0-D array) or ``None`` when gradients are skipped.
    estimator_eig : jnp.ndarray
        Scalar estimate (0-D array) of the expected information gain.
    """

    Q, R, _ = model.other_parameters()

    # Number of particles used at each layer of the nested filter
    N_param = param_ps.shape[0]
    N_state = state_ps_nested.shape[1]
    N_obs_per_parameter = N_state

    # --- Step 1: draw samples for the outer (Gamma) and inner expectations ---
    sampling_key, subkey = random.split(subkey, 2)
    (
        state_ps_outer_expectation,
        obs_ps_outer_expectation,
        state_ps_inner_expectation,
        _
    ) = samples_Expectation(
        sampling_key,
        param_ps,
        state_ps_nested,
        design,
        delta_t
    )
   

    # --- Step 2: reshape observations/states into flattened (param, state) pairs ---
    obs_reshaped = obs_ps_outer_expectation.reshape(N_param * N_obs_per_parameter, -1)
    states_reshaped_outer = state_ps_outer_expectation.reshape(N_param * N_state, -1)
    states_reshaped_inner = state_ps_inner_expectation.reshape(N_param * N_state, -1)


    # --- Mini-batch approximation of the EIG ---
    def mini_batch_eig_body(carry, inputs):
        accum_eig, accum_gradient = carry
        obs_block, state_block = inputs

        log_g_ddotx_mini = _batched_log_likelihood(design,R,obs_block,state_block)
        log_g_ddotx_mini = sanitize_logs(log_g_ddotx_mini)


        log_g_dotx_mini = _batched_log_likelihood(design,R,obs_block,states_reshaped_inner
                                                  ).reshape(N_obs_per_parameter,N_param,N_state,-1)                                                                                       
        log_g_dotx_mini = sanitize_logs(log_g_dotx_mini)


        estimator_logL_mini = logsumexp(log_g_ddotx_mini, axis=1) - jnp.log(N_state)
        estimator_logL_mini = estimator_logL_mini.reshape((N_obs_per_parameter,))
        estimator_logZ_mini = logsumexp(log_g_dotx_mini, axis=(1, 2)) - jnp.log(N_state * N_param)
        estimator_logZ_mini = estimator_logZ_mini.reshape((N_obs_per_parameter,))

        eig_value = jnp.mean(estimator_logL_mini - estimator_logZ_mini)

        gradient_value = jnp.zeros_like(design)
        if gradient_estimation:
            # --- Step 6: evaluate gradients of the likelihood ---
            grad_g_ddotx_mini = _batched_grad_likelihood(design, R,obs_block,state_block)
            grad_g_ddotx_mini = sanitize_gradients(grad_g_ddotx_mini)

            grad_g_dotx_mini = _batched_grad_likelihood(design, R, obs_block, states_reshaped_inner
                                                        ).reshape(N_obs_per_parameter,N_param,N_state,-1)
            grad_g_dotx_mini = sanitize_gradients(grad_g_dotx_mini)

            # --- Step 7: Estimators of the gradient of the likelihood and evidence ---
            estimator_gradL_mini = jnp.mean(grad_g_ddotx_mini, axis=(1))
            estimator_gradZ_mini = jnp.mean(grad_g_dotx_mini, axis=(1, 2))  

            # --- Step 8: estimator of the score function (gradient of log p(y | xi)) ---
            grad_log_g_mini = _batched_grad_log_likelihood(design, R, obs_block, state_block)
            grad_log_g_mini = sanitize_gradients(grad_log_g_mini).reshape((N_obs_per_parameter, -1))

            # --- Step 9: combine estimators into the EIG gradient (scalar) ---
            gradient_terms = (
                (estimator_gradL_mini / jnp.maximum(jnp.exp(estimator_logL_mini)[..., None], 1e-10))
                - (estimator_gradZ_mini /  jnp.maximum(jnp.exp(estimator_logZ_mini)[..., None], 1e-10))
                + (estimator_logL_mini - estimator_logZ_mini)[..., None] * grad_log_g_mini
            )

            gradient_value = jnp.mean(gradient_terms, axis=0)

        new_accum_eig = accum_eig + eig_value
        new_accum_gradient = accum_gradient + gradient_value
        return (new_accum_eig, new_accum_gradient), eig_value

    initial_carry = (
        jnp.asarray(0.0, dtype=design.dtype),
        jnp.asarray(jnp.zeros_like(design), dtype=design.dtype)
    )

    debug_mini_batch = False
    if debug_mini_batch:
        carry = initial_carry
        for idx in range(N_param):
            carry, eig_value = mini_batch_eig_body(
                carry,
                (
                    obs_ps_outer_expectation[idx],
                    state_ps_outer_expectation[idx]
                )
            )

        total_eig_mini, total_gradient_mini = carry
    else:
        # (Optional debug loop above replaces this scan when DEBUG_MINI_BATCH=1)
        (total_eig_mini, total_gradient_mini), _ = lax.scan(
            mini_batch_eig_body,
            initial_carry,
            (obs_ps_outer_expectation, state_ps_outer_expectation)
        )


    estimator_eig_mini = total_eig_mini / N_param

    if gradient_estimation:
        estimator_gradient_eig_mini = total_gradient_mini / N_param
    else:
        estimator_gradient_eig_mini = None

    estimator_gradient_eig = estimator_gradient_eig_mini
    estimator_eig = estimator_eig_mini


    # # --- Step 3: evaluate log-likelihood ---
    # log_likelihood_ddotx = _batched_log_likelihood(
    #     design,
    #     R,
    #     obs_reshaped,
    #     states_reshaped_outer
    # ).reshape(
    #     N_param,
    #     N_obs_per_parameter,
    #     N_param,
    #     N_state
    # )
    # log_likelihood_ddotx = jnp.einsum('ibil->ibl', log_likelihood_ddotx)

    # log_likelihood_dotx = _batched_log_likelihood(
    #     design,
    #     R,
    #     obs_reshaped,
    #     states_reshaped_inner
    # ).reshape(
    #     N_param,
    #     N_obs_per_parameter,
    #     N_param,
    #     N_state
    # )

    # # --- Step 4: Monte-Carlo estimators for log p(y | theta, xi) and log p(y | xi) ---
    # estimator_loglikelihood = logsumexp(log_likelihood_ddotx, axis=2) - jnp.log(N_state)
    # estimator_logevidence = logsumexp(log_likelihood_dotx, axis=(2, 3)) - jnp.log(N_state * N_param)

    # # --- Step 5: expected information gain (scalar) ---
    # estimator_eig = jnp.mean(estimator_loglikelihood - estimator_logevidence)

    # print(f"EIG mini: {estimator_eig_mini}, EIG full: {estimator_eig}")
    
    # if gradient_estimation:

    #     # --- Step 6: evaluate gradients of the likelihood ---
    #     grad_likelihood_ddotx = _batched_grad_likelihood(
    #         design,
    #         R,
    #         obs_reshaped,
    #         states_reshaped_outer
    #     ).reshape(
    #         N_param,
    #         N_obs_per_parameter,
    #         N_param,
    #         N_state,
    #         -1
    #     )
    #     grad_likelihood_ddotx = jnp.einsum('ibilm->iblm', grad_likelihood_ddotx)

    #     grad_likelihood_dotx = _batched_grad_likelihood(
    #         design,
    #         R,
    #         obs_reshaped,
    #         states_reshaped_inner
    #     ).reshape(
    #         N_param,
    #         N_obs_per_parameter,
    #         N_param,
    #         N_state,
    #         -1
    #     )


    #     # --- Step 7: Estimators of the gradient of the likelihood and evidence ---
    #     estimator_grad_likelihood = jnp.mean(
    #         grad_likelihood_ddotx,
    #         axis=(2)
    #     )
    #     estimator_grad_evidence = jnp.mean(
    #         grad_likelihood_dotx,
    #         axis=(2, 3)
    #     )
       

    #     # --- Step 8: estimator of the score function (gradient of log p(y | xi)) ---
    #     grad_log_observation_pdf = _batched_grad_log_likelihood(
    #         design,
    #         R,
    #         obs_ps_outer_expectation.reshape(N_param * N_state, -1),
    #         state_ps_outer_expectation.reshape(N_param * N_state, -1)
    #     ).reshape(
    #         N_param,
    #         N_state,
    #         -1
    #     )
        

    #     # --- Step 9: combine estimators into the EIG gradient (scalar) ---
    #     estimator_gradient_eig = jnp.mean(
    #         (estimator_grad_likelihood / jnp.exp(estimator_loglikelihood)[..., None])
    #         - (estimator_grad_evidence / jnp.exp(estimator_logevidence)[..., None])
    #         + (estimator_loglikelihood - estimator_logevidence)[..., None] * grad_log_observation_pdf
    #     )
       
    # else:
    #     estimator_gradient_eig = None

    # print (f"Gradient EIG mini: {estimator_gradient_eig_mini}, Gradient EIG full: {estimator_gradient_eig}")

    return estimator_gradient_eig, estimator_eig


def optimization_step(
        subkey,
        design,
        delta_t,
        state_ps, 
        param_ps,
        k, 
        m_prev, 
        v_prev
    ):

    # ADAM parameters
    model_type = MODEL_TYPE
    if model_type == 'sir':
        alpha = 0.03
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-6
        max_step = 0.5
    elif model_type == 'source':
        alpha = 0.01
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-6
        max_step = 0.1

    # Reshape design (just to be sure)
    design = design.reshape((model._DESIGN_DIM,))
    
    
    # Compute the gradient of the EIG
    gradient_key, subkey = random.split(subkey, 2)
    gradient_eig, eig = gradient_EIG(
                                    subkey=gradient_key,
                                    param_ps=param_ps,
                                    state_ps_nested=state_ps,
                                    design=design,
                                    delta_t=delta_t,
                                    gradient_estimation=True
                                )
    gradient_eig = gradient_eig.reshape((model._DESIGN_DIM,))

    # ADAM update
    m_aux = beta_1 * m_prev + (1 - beta_1) * gradient_eig
    v_aux = beta_2 * v_prev + (1 - beta_2) * gradient_eig**2
    m_now = m_aux / (1 - beta_1**(k+1))
    v_now = v_aux / (1 - beta_2**(k+1))
    opt_step = alpha * m_now / (jnp.sqrt(v_now) + epsilon)
    opt_step = jnp.clip(opt_step, -max_step, max_step)
    new_design = design + opt_step
    
    return new_design, gradient_eig, eig, m_aux, v_aux


def _run_design_optimization_impl(
        subkey,
        init_design,
        delta_t,
        state_ps_nested,
        param_ps,
        K
    ):

    designs_hist = jnp.zeros((K,model._DESIGN_DIM), dtype=jnp.result_type(K, init_design))
    eig_hist = jnp.zeros((K,), dtype=jnp.result_type(delta_t))

    moment_dtype = init_design.dtype
    m_prev = jnp.zeros_like(init_design, dtype=moment_dtype)
    v_prev = jnp.zeros_like(init_design, dtype=moment_dtype)

    step_keys = random.split(subkey, K)
    loop_keys = step_keys

    def body_fun(k, carry):
        design, m_prev, v_prev, designs_hist, eig_hist = carry
        step_key = loop_keys[k]

        new_design, gradient_eig, eig, m_prev, v_prev = optimization_step(
            subkey=step_key,
            design=design,
            delta_t=delta_t,
            state_ps=state_ps_nested,
            param_ps=param_ps,
            k=k,
            m_prev=m_prev.astype(moment_dtype),
            v_prev=v_prev.astype(moment_dtype)
        )
        jax_debug.print(
            "Iteration {}/{} - Design: {} - sigmoid(design): {} - EIG: {} - Gradient EIG: {}",
            k + 1,
            K,
            new_design,
            sigmoid(new_design),
            eig,
            gradient_eig,
        )

        designs_hist = designs_hist.at[k].set(new_design)
        eig_hist = eig_hist.at[k].set(eig)

        return new_design, m_prev, v_prev, designs_hist, eig_hist

    carry0 = (
        init_design,
        m_prev,
        v_prev,
        designs_hist,
        eig_hist
    )
    
    final_design, m_prev, v_prev, designs_hist, eig_hist = lax.fori_loop(
        0, K, body_fun, carry0
    )

    best_design = final_design
    best_eig = eig_hist[-1]
    return best_design, best_eig, designs_hist, eig_hist


_run_design_optimization_jit = partial(jit, static_argnums=(5))(_run_design_optimization_impl)


def _run_design_optimization(
        subkey,
        init_design,
        delta_t,
        state_ps_nested,
        param_ps,
        K
    ):
    results = _run_design_optimization_jit(
        subkey,
        init_design,
        delta_t,
        state_ps_nested,
        param_ps,
        K
    )
    best_design, best_eig, designs_hist, eig_hist = results
    return best_design, best_eig, designs_hist, eig_hist


def main(
        seed_run=0, 
        delta_t=0.05,
        T_run=2, 
        N_state=20, 
        N_param=20,
        verbose=True,
        save_results=False,
        fixed_design=False,
        random_design=False,
        use_wandb=True,
        K_opt=50
    ):
   
    time_start = time.time()
   
    # Set up random key
    key = random.PRNGKey(seed_run)
    subkey, _ = random.split(key)

    # Setting wandb run
    if use_wandb:
        
        # Set the run name based on the model type and parameters
        model_type = MODEL_TYPE
        if model_type == 'sir':
            run = wandb.init(
                entity="sarapv-aalto-university",
                project="aistats-sirmodel", 
                config={
                    "seed": seed_run,
                    "T": T_run,
                    "delta_t": delta_t,
                    "N_state": N_state,
                    "N_param": N_param
                }
            )
            if random_design:
                wandb.run.name = f"sir_random_design_run_T{T_run}_Ns{N_state}_Np{N_param}_K{K_opt}_seed{seed_run}"
            elif fixed_design:
                wandb.run.name = f"sir_fixed_design_run_T{T_run}_Ns{N_state}_Np{N_param}_K{K_opt}_seed{seed_run}"
            else:
                wandb.run.name = f"sir_run_T{T_run}_Ns{N_state}_Np{N_param}_K{K_opt}_seed{seed_run}"
        elif model_type == 'source':
            run = wandb.init(
                entity="sarapv-aalto-university",
                project="aistats-sourcemodel", 
                config={
                    "seed": seed_run,
                    "T": T_run,
                    "delta_t": delta_t,
                    "N_state": N_state,
                    "N_param": N_param
                }
            )
            if random_design:
                wandb.run.name = f"source_random_design_run_T{T_run}_Ns{N_state}_Np{N_param}_K{K_opt}_seed{seed_run}"
            elif fixed_design:
                wandb.run.name = f"source_fixed_design_run_T{T_run}_Ns{N_state}_Np{N_param}_K{K_opt}_seed{seed_run}"
            else:
                wandb.run.name = f"source_run_T{T_run}_Ns{N_state}_Np{N_param}_K{K_opt}_seed{seed_run}"
        else:
            model_type = 'sir'
            run = wandb.init(
                entity="sarapv-aalto-university",
                project="aistats-sirmodel", 
                config={
                    "seed": seed_run,
                    "T": T_run,
                    "delta_t": delta_t,
                    "N_state": N_state,
                    "N_param": N_param
                }
            )
            wandb.run.name = f"sir_run_T{T_run}_Ns{N_state}_Np{N_param}_K{K_opt}_seed{seed_run}"
    
    # Create results directory if saving results
    results_dir = None
    if save_results:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = MODEL_TYPE
        if model_type == 'sir':
            results_dir = f'./results/sir/experiment_{timestamp}'
        elif model_type == 'source':
            results_dir = f'./results/source/experiment_{timestamp}'
        else:
            results_dir = f'./results/experiment_{timestamp}'
        os.makedirs(results_dir, exist_ok=True)

    
    true_param = model.true_parameters()

    # Run NPF algorithm
    print(f"Running NPF with {N_param} parameter particles and {N_state} state particles for {T_run} time steps")
    
    # Set up progress bar if verbose
    if verbose:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=T_run, desc="NPF Progress")
        except ImportError:
            print("tqdm not available, not showing progress bar")
            pbar = None
    else:
        pbar = None
    
    
    parameter_estimates_history, state_estimates_history, last_design, true_states_history = NPF_param_layer(
                                                                            subkey=subkey,
                                                                            true_parameters=true_param,
                                                                            delta_t=delta_t,
                                                                            T_run=T_run,
                                                                            N_state=N_state,
                                                                            N_param=N_param,
                                                                            pbar=pbar,
                                                                            use_wandb=use_wandb,
                                                                            save_results=save_results,
                                                                            results_dir=results_dir,
                                                                            random_design=random_design,
                                                                            fixed_design=fixed_design,
                                                                            K_opt=K_opt
                                                                        )
                                                                        

    time_end = time.time()
    print(f"Total execution time: {time_end - time_start}")

    if save_results:
        # Save configuration and execution time as text file for reference
        config_path = os.path.join(results_dir, 'config.txt')
        with open(config_path, 'w') as f:
            f.write(f"# Experiment Configuration\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Seed: {seed_run}\n")
            f.write(f"True parameters: {true_param}\n")
            f.write(f"Time steps (T): {T_run}\n")
            f.write(f"Delta t: {delta_t}\n")
            f.write(f"N_state: {N_state}\n")
            f.write(f"N_param: {N_param}\n")
            f.write(f"Model type: {MODEL_TYPE}\n")
            f.write(f"Execution time (s): {time_end - time_start}\n")
            
        if verbose:
            print(f"Configuration saved to {config_path}")
    
    if verbose:
        print("Final parameter estimates:", parameter_estimates_history[-2])
        print("True parameters:", true_param)

    if use_wandb:
        # Time axis
        t_axis = np.arange(T_run)  # exclude last time step

        # Plot and log true vs estimated states
        if MODEL_TYPE == 'sir':
            fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
            labels = [("S1", 0), ("I1", 1), ("S2", 2), ("I2", 3)]
            true_np = np.asarray(true_states_history)[:-1]
            est_np = np.asarray(state_estimates_history)[:-1]
            for ax, (name, idx) in zip(axs.flat, labels):
                ax.plot(t_axis, true_np[:, idx], label=f"true {name}")
                ax.plot(t_axis, est_np[:, idx], linestyle="--", label=f"est {name}")
                ax.set_title(name)
                ax.grid(True, alpha=0.3)
            axs[1,0].set_xlabel("time")
            axs[1,1].set_xlabel("time")
            axs[0,0].legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            wandb.log({"states_vs_estimates": wandb.Image(fig)})
            plt.close(fig)

        elif MODEL_TYPE == 'source':
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            true_np = np.asarray(true_states_history)[:-1]
            est_np = np.asarray(state_estimates_history)[:-1]
            names = [("p_x", 0), ("p_y", 1), ("phi", 2)]
            for ax, (name, idx) in zip(axs, names):
                ax.plot(t_axis, true_np[:, idx], label=f"true {name}")
                ax.plot(t_axis, est_np[:, idx], linestyle="--", label=f"est {name}")
                ax.set_ylabel(name)
                ax.grid(True, alpha=0.3)
            axs[-1].set_xlabel("time")
            axs[0].legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            wandb.log({"states_vs_estimates": wandb.Image(fig)})
            plt.close(fig)

            # 2D trajectory plot (x-y) for source model
            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
            # true and estimated positions over time (exclude last step already applied above)
            ax2.plot(true_np[:, 0], true_np[:, 1], label="true trajectory")
            ax2.plot(est_np[:, 0], est_np[:, 1], linestyle="--", label="est trajectory")
            # plot fixed sensor locations as triangles, and their final-step orientation as arrows (optional)
            try:
                sensors = np.asarray(model._SENSORS)
                # plot sensors
                ax2.scatter(sensors[:, 0], sensors[:, 1], marker="^", color="black", s=60, label="sensors")
                # draw orientation arrows using the last design (if available)
                try:
                    last_angles = last_design      # angles for each sensor at final time step
                    L = 0.6                       # arrow length
                    for j, (sx, sy) in enumerate(sensors):
                        ang = float(last_angles[j])
                        ax2.arrow(sx, sy, L*np.cos(ang), L*np.sin(ang),
                                  head_width=0.15, head_length=0.2, fc="gray", ec="gray", alpha=0.8)
                except Exception:
                    # if designs not available or shapes mismatch, skip arrows gracefully
                    pass
            except Exception:
                # if sensor positions are unavailable, skip sensor plotting
                pass
            # mark start/end of true trajectory
            ax2.scatter(true_np[0, 0], true_np[0, 1], marker="o", s=40, label="start (true)")
            ax2.scatter(true_np[-1, 0], true_np[-1, 1], marker="x", s=50, label="end (true)")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_aspect("equal", adjustable="box")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best", fontsize=8)
            fig2.tight_layout()
            wandb.log({"trajectory_2d": wandb.Image(fig2)})
            plt.close(fig2)

            pe = np.asarray(parameter_estimates_history)[:-1]
            if pe.size > 0:
                tp = np.asarray(true_param)
                err = pe - tp  # signed error over time
                param_dim = err.shape[1]

                label_map = {
                    'sir': [r"$\beta_1$", r"$\gamma_1$", r"$\beta_2$", r"$\gamma_2$"],
                    'source': [r"$v_x$", r"$v_y$"],
                }
                fallback_labels = [fr"$\theta_{i+1}$" for i in range(param_dim)]
                labels = label_map.get(MODEL_TYPE, fallback_labels)
                if len(labels) < param_dim:
                    labels = (labels + fallback_labels)[ : param_dim]
                fig3, axs3 = plt.subplots(param_dim, 1, figsize=(10, 2.5 * param_dim), sharex=True)
                axs3 = np.atleast_1d(axs3)

                for i in range(param_dim):
                    ax = axs3[i]
                    ax.plot(t_axis, err[:, i])
                    ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.6)
                    ax.set_ylabel(f"{labels[i]} error")
                    ax.grid(True, alpha=0.3)
                axs3[-1].set_xlabel("time")
                fig3.tight_layout()
                wandb.log({"parameter_errors": wandb.Image(fig3)})
                plt.close(fig3)

        # --- Time-series RMSE plots (over steps 0..T-1) ---
        # State RMSE per step (average over state dims)
        state_err = np.asarray(state_estimates_history)[:-1] - np.asarray(true_states_history)[:-1]  # (T, state_dim)
        rmse_state_time = np.sqrt(np.mean(state_err**2, axis=1))               # (T,)

        # Parameter RMSE per step (average over parameter dims)
        param_err = np.asarray(parameter_estimates_history)[:-1] - np.asarray(true_param)[None, :]  # (T, param_dim)
        rmse_param_time = np.sqrt(np.mean(param_err**2, axis=1))                             # (T,)

        # Plot RMSE time series
        fig_rmse, ax_rmse = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax_rmse[0].plot(t_axis, rmse_state_time)
        ax_rmse[0].set_ylabel("state RMSE")
        ax_rmse[0].grid(True, alpha=0.3)
        ax_rmse[1].plot(t_axis, rmse_param_time)
        ax_rmse[1].set_ylabel("parameter RMSE")
        ax_rmse[1].set_xlabel("time")
        ax_rmse[1].grid(True, alpha=0.3)
        fig_rmse.tight_layout()
        wandb.log({"rmse_time_series": wandb.Image(fig_rmse)})
        plt.close(fig_rmse)

    

        run.finish()
        
    results = {
        "parameter_estimates_history": parameter_estimates_history,
        "state_estimates_history": state_estimates_history,
        "true_states_history": true_states_history,
        "execution_time": time_end - time_start,
        "results_dir": results_dir
    }

    return results


if __name__ == "__main__":
    # This block will be executed when the script is run directly
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Bayesian adaptive design for partially observable dynamical systems.')
    parser.add_argument('--seed_run', type=int, default=0, help='Random seed')
    parser.add_argument('--randomd', action='store_true', dest='random_design', help='Use random design')
    parser.add_argument('--fixedd', action='store_true', dest='fixed_design', help='Use fixed design')
    parser.add_argument('--T_run', type=int, default=2, help='Number of time steps')
    parser.add_argument('--N_state', type=int, default=50, help='Number of state particles')
    parser.add_argument('--N_param', type=int, default=52, help='Number of parameter particles')
    parser.add_argument('--delta_t', type=float, default=0.05, help='Time step')
    # parser.add_argument('--no_history', action='store_false', dest='save_history', 
    #                    help='Disable saving particle history')
    parser.add_argument('--quiet', action='store_false', dest='verbose',
                       help='Run quietly without progress information')
    parser.add_argument('--save', action='store_true', dest='save_results',
                       help='Enable saving results to disk')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--K_opt', type=int, default=50,
                        help='Number of optimisation iterations for design updates when optimisation is enabled')
    args = parser.parse_args()

    
    # Run with command line arguments
    results = main(
        seed_run=args.seed_run,
        random_design=args.random_design,
        fixed_design=args.fixed_design,
        delta_t=args.delta_t,
        T_run=args.T_run,
        N_state=args.N_state,
        N_param=args.N_param,
        verbose=args.verbose,
        save_results=args.save_results,
        use_wandb=args.use_wandb,
        K_opt=args.K_opt
    )
