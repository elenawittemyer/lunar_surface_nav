import jax
import jax.numpy as np
from jax import grad, jacfwd, vmap, jit, hessian
from hillshade import get_shadow_map_stack
import cv2
import time

def convert_pos(pos_array, size):
    return np.round(pos_array-size/2)

time_horizon = 100
x_0 = np.array([[5, 0], [7, 0], [3, 0]])
x_f = np.array([[-10, 0], [-10, 0], [-10, 0]])
x = np.linspace(x_0, x_f, time_horizon, endpoint=True)
u = np.zeros((time_horizon, 3, 2))

path = "DEMs/Site01_final_adj_5mpp_surf.tif"
shadow_map_stack = get_shadow_map_stack(path, 'Site01', start_time=0, end_time=20000, dt=200)
shadows_idx_stack = []
for i in range(len(shadow_map_stack)):
    shadow_map = shadow_map_stack[i]
    resized_x = shadow_map.shape[1] // 10
    resized_y = shadow_map.shape[0] // 10
    resized_shadow_map = cv2.resize(shadow_map, (resized_x, resized_y), interpolation=cv2.INTER_AREA)
    shadow_idx = np.where(resized_shadow_map<40)
    shadow_idx_array = 10*np.array([shadow_idx[1], shadow_idx[0]]).T
    shadow_idx_array = convert_pos(shadow_idx_array, np.shape(shadow_map)[0])
    shadows_idx_stack.append(shadow_idx_array)

def padding(shadow_map, max_len, map_size):
    current_len = shadow_map.shape[0]
    padded_vals = map_size*np.ones((max_len-current_len, 2))
    return np.vstack((shadow_map, padded_vals))

map_size =  np.shape(shadow_map_stack[0])[0]
max_len = max(arr.shape[0] for arr in shadows_idx_stack)
for i in range(len(shadows_idx_stack)):
    if len(shadows_idx_stack[i])<max_len:
        shadows_idx_stack[i] =  padding(shadows_idx_stack[i], max_len, map_size)
shadows_idx_stack = np.array(shadows_idx_stack)

start_time = time.time()

control_constraint =  abs(u)-5
def dist_to_shadow(obstacle, x_t):
    dist = np.linalg.norm(x_t - obstacle, axis=1).flatten()
    constraint_vals = 10 - dist
    return constraint_vals
def dist_t(shadow_t, x_t):
    shadow_constraint_t_k = vmap(dist_to_shadow, in_axes=(0, None))(shadow_t, x_t)
    return shadow_constraint_t_k

idx = np.linspace(0, len(x), 100).astype(int)

def get_shadow_constraint_t(shadow_t, x):
    shadow_constraint_t = vmap(dist_t, in_axes=(None, 0))(shadow_t, x)
    return shadow_constraint_t

overkill_shadow_constraint = vmap(get_shadow_constraint_t, in_axes=(0, None))(shadows_idx_stack, x) 

def get_slice(a_i, i):
    return a_i[i, :, :]  # Select the i-th (156, 3) slice from a_i of shape (N, 156, 3)

shadow_constraint = jax.vmap(get_slice)(overkill_shadow_constraint, np.arange(time_horizon))
_g = np.append(.1*shadow_constraint.flatten(), control_constraint)
print(_g)
