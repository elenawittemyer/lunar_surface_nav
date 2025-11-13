import jax.numpy as np
from jax import vmap
import numpy as onp
import cv2
from multi_rover_ergodic.erg_expl import ErgodicTrajectoryOpt
from multi_rover_ergodic.gaussian import gaussian
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from hillshade import get_shadow_map, get_shadow_map_stack
from matplotlib.animation import FuncAnimation, FFMpegWriter
from info_distrib import random_info

def main(num_agents, map_size, time_args, info_map=None, init_pos=None, plot=True, shadows = None, craters = None):
    if info_map is None:
        info_map = sample_map(map_size)
    if init_pos is None:
        init_pos = sample_initpos(num_agents, map_size)
        
    path_travelled = np.empty(shape=(num_agents, 2) + (0, )).tolist()

    traj_opt = ErgodicTrajectoryOpt(init_pos, info_map, num_agents, map_size, shadows, craters, time_args)
    for k in range(100):
        traj_opt.solver.solve(max_iter=1000)
        sol = traj_opt.solver.get_solution()
        clear_output(wait=True)

    for i in range(num_agents):
        path_travelled[i][0].append(sol['x'][:,i][:,0]+(map_size/2))
        path_travelled[i][1].append(sol['x'][:,i][:,1]+(map_size/2))

    np.save('path_data.npy', path_travelled)
    if plot == True:
        cmap = get_colormap(num_agents+1)
        fig, ax = plt.subplots()
        ax.imshow(info_map, cmap='Greys_r', origin="upper")
        for i in range(num_agents):
            ax.plot(np.array(path_travelled[i][0]).flatten(), np.array(path_travelled[i][1]).flatten(),  c=cmap(i), label='Agent ' + str(i + 1))
        plt.legend(bbox_to_anchor=(1.2, 1.1), loc='upper right', framealpha=1)

        
        '''
        if shadows!=None:
            obstacles = np.round(shadows + map_size/2)
            for obstacle in obstacles:
                circle = patches.Circle([obstacle[0], obstacle[1]], radius=10)
                ax.add_patch(circle)
        '''

        plt.show()

    return path_travelled

### Helpers ###########################################

def sample_map(size, peaks=3):
    pos = np.floor(onp.random.uniform(0, size, 2*peaks))
    pmap = gaussian(size, pos[0], pos[1], 10)
    peak_indices = [np.where(pmap>.1)]
    for i in range(1, peaks):
        new_peak = gaussian(size, pos[2*i], pos[2*i+1], 10)
        pmap += gaussian(size, pos[2*i], pos[2*i+1], 10)
        peak_indices.append(np.where(new_peak>.1))
    return pmap

def sample_initpos(agents, size):
    return onp.random.uniform(-size/2, size/2, (agents, 2))

def get_colormap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

def convert_pos(pos_array, size):
    return np.round(pos_array-size/2)

def obstacle_pos(size):
    obstacle_coords = np.array([[100, 100], [150, 150]])
    obstacle_coords = convert_pos(obstacle_coords, size)
    return obstacle_coords


def get_shadow_stack(path, time_args, bounds):
    
    time_horizon = time_args['time_horizon']
    start_time = time_args['start_time']
    end_time = time_args['end_time']
    dt = time_args['dt']

    shadow_map_stack = get_shadow_map_stack(path, 'Site01', bounds, start_time, end_time, dt)
    
    if ((end_time-start_time)//dt)!=time_horizon:
        raise Exception('Time horizon and number of time steps do not match.')
    
    shadows_idx_stack = []
    for i in range(len(shadow_map_stack)):
        scale = 10
        shadow_map = shadow_map_stack[i]
        #bounds_frac = [(bounds[0,1]-bounds[0,0])/shadow_map.shape[1], (bounds[1,1]-bounds[1,0])/shadow_map.shape[0]]
        resized_x = shadow_map.shape[1] // scale
        resized_y = shadow_map.shape[0] // scale
        resized_shadow_map = cv2.resize(shadow_map, (resized_x, resized_y), interpolation=cv2.INTER_AREA)
        shadow_idx = np.where(resized_shadow_map<40)
        shadow_idx_array = scale*np.array([shadow_idx[1], shadow_idx[0]]).T
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

    return shadow_map_stack, shadows_idx_stack

def illuminated_craters(crater_pos_arr, shadow_stack, size):
    landmark_idx = []
    for i in range(len(shadow_stack)):
        crater_pos_idx = convert_pos(crater_pos_arr, size).astype(int)
        crater_pos_tuple = tuple((crater_pos_idx[:,0], crater_pos_idx[:,1]))
        crater_light_vals = shadow_stack[i][crater_pos_tuple]
        landmark_pos = crater_pos_idx[np.where(crater_light_vals>50)]
        landmark_idx.append(crater_pos_idx[np.where(crater_light_vals>50)])

    return np.array(landmark_idx)
    #TODO: this doesn't work because the landmark idx values are not all the same size so an array can't be formed (for jit, they all have to be the same size).
    # need to consider what to set 'landmark_idx' to when neither craters are illuminated since this idx is associated with a cost


def animate_plot(path_travelled, num_agents, time_horizon, craters, pmap):
    cmap = get_colormap(num_agents+1)
    pos_x = []
    pos_y = []
    for i in range(num_agents):
        pos_x.append(np.array(path_travelled[i][0]).flatten()) 
        pos_y.append(np.array(path_travelled[i][1]).flatten())

    fig, ax = plt.subplots()
    img = ax.imshow(shadow_map_stack[0], cmap='Greys_r', origin='upper', animated = True)
    overlay = ax.imshow(pmap, origin='upper', animated = True)
    
    for crater in craters:
        circle = patches.Circle([crater[0], crater[1]], radius=10)
        ax.add_patch(circle)
    

    for i in range(num_agents):
        line = [[pos_x[i][0], pos_x[i][1]], [pos_y[i][0], pos_y[i][1]]]
        traj, = ax.plot(line[0], line[1], c=cmap(i))

    
    def updatefig(frame, img, traj, ax):
        img.set_array(shadow_map_stack[frame])
        overlay.set_array(pmap)
        for i in range(num_agents):
            line = [[pos_x[i][frame], pos_x[i][frame+1]], [pos_y[i][frame], pos_y[i][frame+1]]]
            traj, = ax.plot(line[0],line[1], c=cmap(i))
        return img, traj

    ani = FuncAnimation(fig, updatefig, frames=time_horizon, fargs=(img, traj, ax), blit=True)
    FFwriter = FFMpegWriter(fps=10, codec='libx264', bitrate=1800)
    ani.save('shadow_avoidance.mp4', writer=FFwriter)

#######################################################

dem_path = "DEMs/Site01_final_adj_5mpp_surf.tif"
time_args = {
    'dt': 200,
    'start_time': 0,
    'end_time': 20000,
    'time_horizon': 100
}

shadow_map_stack, shadow_idx_stack = get_shadow_stack(dem_path, time_args, bounds=np.array([[0, 1000], [0, 1000]]))
shadow_map = shadow_map_stack[0] #TODO: update info map to change over time
size = np.shape(shadow_map)[0]
crater_pos = np.array([[87, 168], [44, 56], [92, 183]])
init_pos = convert_pos(np.array([[280, 50], [15, 125], [130, 185]]), np.shape(shadow_map)[0])

pmap = random_info(size)


main(num_agents = 3, map_size = size, time_args = time_args, init_pos = init_pos, info_map = pmap, shadows = shadow_idx_stack, craters = crater_pos)
path_travelled = np.load('path_data.npy')
animate_plot(path_travelled, 3, 100, crater_pos, pmap)


'''
crater_idx = illuminated_craters(crater_pos_arr = np.array([[50, 30], [180, 200]]), shadow_stack = shadow_map_stack, size = size)
crater_data = {
    'idx' : crater_idx,
    'rad' : np.array([10, 10])
}
'''