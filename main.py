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

def main(num_agents, map_size, time_args, info_map=None, pos=None, plot=True, shadows = None, craters = None):
    if info_map is None:
        info_map = sample_map(map_size) #fix
    if pos is None:
        init_pos = sample_initpos(num_agents, map_size) #fix
        pos = [init_pos, init_pos]
        
    path_travelled = np.empty(shape=(num_agents, 2) + (0, )).tolist()

    traj_opt = ErgodicTrajectoryOpt(pos, info_map, num_agents, map_size, shadows, craters, time_args)
    for k in range(100):
        traj_opt.solver.solve(max_iter=1000)
        sol = traj_opt.solver.get_solution()
        clear_output(wait=True)

    for i in range(num_agents):
        path_travelled[i][0].append(sol['x'][:,i][:,0]+(map_size[1]/2))
        path_travelled[i][1].append(sol['x'][:,i][:,1]+(map_size[0]/2))

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
    x_conv = np.round(pos_array[:,0]-size[1]/2)
    y_conv = np.round(pos_array[:,1]-size[0]/2)
    return np.vstack((x_conv, y_conv)).T #TODO: fix this

def obstacle_pos(size):
    obstacle_coords = np.array([[100, 100], [150, 150]])
    obstacle_coords = convert_pos(obstacle_coords, size)
    return obstacle_coords

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


def animate_plot(path_travelled, num_agents, time_args, pmap):
    time_horizon = time_args['time_horizon']
    total_time = time_args['end_time'] - time_args['start_time']
    extent = [0, 16000, 0, 16000]
    fps = 10
    size = pmap.shape[0]
    cmap = get_colormap(num_agents+1)
    pos_x = []
    pos_y = []
    for i in range(num_agents):
        pos_x.append(np.array(path_travelled[i][0]).flatten()) 
        pos_y.append(np.array(path_travelled[i][1]).flatten())

    fig, ax = plt.subplots()
    img = ax.imshow(shadow_map_stack[0], cmap='Greys_r', origin='upper', animated = True)
    num_ticks = len(ax.get_xticks())
    tick_labels = np.linspace(0, 16, num_ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    plt.xlabel('km')
    plt.ylabel('km')
    #plt.title('Timescale x' + str(total_time/(time_horizon/fps)))
    overlay = ax.imshow(pmap, origin='upper', alpha = .5, animated = True)
    cbar = plt.colorbar(overlay)
    cbar.ax.set_title('Information Density')
    
    '''
    for crater in craters:
        circle = patches.Circle([crater[0], crater[1]], radius=10)
        ax.add_patch(circle)
    '''

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
    FFwriter = FFMpegWriter(fps=fps, codec='libx264', bitrate=1800)
    ani.save('shadow_avoidance.mp4', writer=FFwriter)

#######################################################

dem_path = "DEMs/Site01_final_adj_5mpp_surf.tif"
time_args = {
    'dt': 100,
    'start_time': 0,
    'end_time': 10000,
    'time_horizon': 100
}

shadow_map_stack, shadow_idx_stack = get_shadow_stack(dem_path, time_args)
shadow_map = shadow_map_stack[0] #TODO: update info map to change over time
size = [np.shape(shadow_map)[0], np.shape(shadow_map)[1]]
#crater_pos = np.array([[87, 168], [44, 56], [92, 183]])
#init_pos = convert_pos(crater_pos, size)

start_pos = np.array([[87, 168], [44, 56], [283, 276]])
end_pos  = np.array([[150,150], [150, 150], [150, 150]])
init_pos = convert_pos(start_pos, size)
final_pos = convert_pos(end_pos, size)

'''
init_pos = np.array([[50,200], [100, 300], [250, 200]])
final_pos = np.array([[300,280], [120, 60], [280, 50]])
init_pos = convert_pos(init_pos, size)
final_pos = convert_pos(final_pos, size)
'''
startstop = [init_pos, final_pos]

pmap = random_info(size)
#pmap = np.ones((size, size))

main(num_agents = 3, map_size = size, time_args = time_args, pos = startstop, info_map = pmap, shadows = shadow_idx_stack, craters=None)
path_travelled = np.load('path_data.npy')
animate_plot(path_travelled, 3, time_args, pmap)

#TODO: fix all the size issues on this page and within ErgodicTrajectoryOpt. convertpos() has already been corrected