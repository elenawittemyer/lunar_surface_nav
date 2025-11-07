import jax
from functools import partial
from jax import grad, jacfwd, vmap, jit, hessian
from jax.lax import scan
import jax.random as jnp_random
import jax.numpy as np

from jax.flatten_util import ravel_pytree

import numpy as onp
from .opt_solver import AugmentedLagrangian
from .dynamics import SingleIntegrator
from .ergodic_metric import ErgodicMetric
from .utils import BasisFunc, get_phik, get_ck
from .target_distribution import TargetDistribution
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
    
def convert_pos(pos_array, size):
    return np.round(pos_array-size/2)
"""Luke waz here"""
class ErgodicTrajectoryOpt(object):
    def __init__(self, initpos, pmap, num_agents, size, shadows, craters, time_args) -> None:
        time_horizon = time_args['time_horizon']
        time_step = time_args['dt']
        self.basis           = BasisFunc(n_basis=[5,5])
        self.erg_metric      = ErgodicMetric(self.basis)
        self.robot_model     = SingleIntegrator(num_agents, time_step)
        n,m,N = self.robot_model.n, self.robot_model.m, self.robot_model.N
        self.target_distr    = TargetDistribution(pmap, size)
        opt_args = {
            'x0' : initpos,
            'xf' : np.array([[-10, 0], [-10, 0], [-10, 0]]),
            'phik' : get_phik(self.target_distr.evals, self.basis)
        }
        ''' Initialize state '''
        x = np.linspace(opt_args['x0'], opt_args['xf'], time_horizon, endpoint=True)
        u = np.zeros((time_horizon, N, m))
        self.init_sol = np.concatenate([x, u], axis=2) 
        def _emap(x):
            ''' Map state space to exploration space '''
            return np.array([(x+(size/2))/size])
        emap = vmap(_emap, in_axes=0)

        def barrier_cost(e):
            """ Barrier function to avoid robot going out of workspace """
            return (np.maximum(0, e-1) + np.maximum(0, -e))**2
        
        def shadow_cost(x, pmap):
            pmap_local = np.array(pmap)
            x_coords = np.reshape(x, (time_horizon*num_agents, 2))
            row_idx = np.clip((x_coords[:,0] + size/2).astype(int), 0, pmap.shape[0]-1)
            col_idx = np.clip((x_coords[:,1] + size/2).astype(int), 0, pmap.shape[0]-1)
            def get_value(row, col):
                return pmap_local[row, col]
            pmap_vals = vmap(get_value)(row_idx, col_idx)
            return np.sum(100-pmap_vals)
        
        def landmark_dist_penalty(x):
            crater_idx = craters['idx']
            crater_rad = craters['rad']
            def total_crater_dist(x, crater):
                def single_crater_dist(x, crater_i):
                    def single_crater_dist_t(x_t, crater_t):
                        return np.linalg.norm(x_t-crater_t)
                    return vmap(single_crater_dist_t, in_axes=(0, None))(x, crater_i)
                return vmap(single_crater_dist, in_axes = (None, 0))(x, crater)
            overkill_penalty = vmap(total_crater_dist, in_axes=(None, 0))(x, crater_idx)
            
            def get_slice(a_i, i):
                return a_i[:, i]

            penalty = vmap(get_slice)(overkill_penalty, np.arange(time_horizon))
            return np.sum(penalty)

        @jit
        def loss(z, args):
            """ Traj opt loss function, not the same as erg metric """
            x, u = z[:, :, :n], z[:, :, n:]
            phik = args['phik']
            e = np.squeeze(emap(x))
            ck = np.mean(vmap(get_ck, in_axes=(1, None))(e, self.basis), axis=0)
            erg_m = self.erg_metric(ck, phik)
            return 10000 * erg_m \
                    + np.mean(u**2) \
                    + np.sum(barrier_cost(e)) \
                    #+ landmark_dist_penalty(x)
                    #+ shadow_cost(x, pmap)
        def eq_constr(z, args):
            """ dynamic equality constriants """
            x, u = z[:, :, :n], z[:, :, n:]
            x0 = args['x0']
            xf = args['xf']
            return np.concatenate([
                (x[0]-x0).flatten(), 
                (x[1:,:]-vmap(self.robot_model.f)(x[:-1,:], u[:-1,:])).flatten(),
                (x[-1] - xf).flatten()
            ])

        def ineq_constr(z,args):
            """ control inequality constraints"""
            x, u = z[:, :, :n], z[:, :, n:]
            control_constraint =  abs(u)-1.
            
            def dist_to_shadow(obstacle, x_t):
                dist = np.linalg.norm(x_t - obstacle, axis=1).flatten()
                constraint_vals = 10 - dist
                return constraint_vals
            
            def dist_t(shadow_t, x_t):
                shadow_constraint_t_k = vmap(dist_to_shadow, in_axes=(0, None))(shadow_t, x_t)
                return shadow_constraint_t_k
            
            def get_shadow_constraint_t(shadow_t, x):
                shadow_constraint_t = vmap(dist_t, in_axes=(None, 0))(shadow_t, x)
                return shadow_constraint_t

            overkill_shadow_constraint = vmap(get_shadow_constraint_t, in_axes=(0, None))(shadows, x) 

            def get_slice(a_i, i):
                return a_i[i, :, :]

            shadow_constraint = vmap(get_slice)(overkill_shadow_constraint, np.arange(time_horizon))
            def step_diff(x):
                diff = np.linalg.norm(x[1:]-x[0:-1], axis = 1)
                return diff
            x_arg = np.transpose(x, (1, 0, 2))
            step_constr = vmap(step_diff)(x_arg)
            upper_step_constr = step_constr - 10
            #lower_step_constr = 1 - step_constr
            
            _g = np.concatenate((.1*shadow_constraint.flatten(), control_constraint.flatten(), upper_step_constr.flatten()))
            
            return _g
        

        self.solver = AugmentedLagrangian(
                                            self.init_sol,
                                            loss, 
                                            eq_constr, 
                                            ineq_constr,
                                            opt_args, 
                                            step_size=0.01,
                                            c=1.0
                    )
        # self.solver.solve()