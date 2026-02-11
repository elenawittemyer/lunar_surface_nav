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
    
class ErgodicTrajectoryOpt(object):
    def __init__(self, pos, pmap, num_agents, size, shadows, time_args) -> None:
        time_horizon = time_args['time_horizon']
        time_step = time_args['dt']
        self.basis           = BasisFunc(n_basis=[5,5])
        self.erg_metric      = ErgodicMetric(self.basis)
        self.robot_model     = SingleIntegrator(num_agents, time_step)
        n,m,N = self.robot_model.n, self.robot_model.m, self.robot_model.N
        self.target_distr    = TargetDistribution(pmap, size)
        opt_args = {
            'x0' : pos[0],
            'xf' : pos[1],
            'phik' : get_phik(self.target_distr.evals, self.basis)
        }
        ''' Initialize state '''
        x = np.linspace(opt_args['x0'], opt_args['xf'], time_horizon, endpoint=True)
        u = np.zeros((time_horizon, N, m))
        self.init_sol = np.concatenate([x, u], axis=2) 

        def _emap(x):
            ''' Map state space to exploration space '''
            width_height = np.flip(np.array(size))
            return np.array([(x+(np.array(width_height)/2))/np.array(width_height)])
        emap = vmap(_emap, in_axes=0)

        def barrier_cost(e):
            """ Barrier function to avoid robot going out of workspace """
            return (np.maximum(0, e-1) + np.maximum(0, -e))**2

        @jit
        def loss(z, args):
            """ Traj opt loss function, not the same as erg metric """
            x, u = z[:, :, :n], z[:, :, n:]
            phik = args['phik']
            e = np.squeeze(emap(x))
            ck = np.mean(vmap(get_ck, in_axes=(1, None))(e, self.basis), axis=0)
            erg_m = self.erg_metric(ck, phik)
            return 1000 * erg_m \
                    + np.mean(u**2) \
                    + np.sum(barrier_cost(e)) 
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

        def idx_to_world(idx, size):
            #idx = [row, col]
            return np.array([
                idx[1] - size[1] / 2, #x
                idx[0] - size[0] / 2  #y
                ])
        
        shadows_t_s = lambda s:idx_to_world(s, size)
        shadows_t = lambda t:vmap(shadows_t_s)(t)
        shadows_world = vmap(shadows_t)(shadows)
        
        def ineq_constr(z,args):
            """ control inequality constraints"""
            x, u = z[:, :, :n], z[:, :, n:]
            control_constraint =  abs(u)-5.

            def shadow_constraint_t(shadow_t, x_t):
                '''
                min_dist_sq = np.sum((x_t[:,None,:] - shadow_t[None,:,:])**2, axis=-1)
                return np.maximum(20.0**2 - min_dist_sq, 0)
                '''
                min_dist_sq = np.min(np.sum((x_t[:,None,:] - shadow_t[None,:,:])**2, axis=-1),axis=1)
                return np.maximum(5.0**2 - min_dist_sq, 0)
                
                '''
                dist_sq = vmap(lambda obs: np.sum((x_t - obs)**2, axis=1))(shadow_t)
                return np.maximum(5.0**2 - dist_sq, 0)
                '''
            shadow_constraint = vmap(shadow_constraint_t)(shadows_world, x)

            def step_diff(x):
                diff = np.linalg.norm(x[1:]-x[0:-1], axis = 1)
                return diff
            x_arg = np.transpose(x, (1, 0, 2))
            step_constr = vmap(step_diff)(x_arg)
            upper_step_constr = step_constr - 10
            #lower_step_constr = 1 - step_constr
            
            sc_weight = 1
            _g = np.concatenate((sc_weight*shadow_constraint.flatten(), control_constraint.flatten(), upper_step_constr.flatten()))
            
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