import jax.numpy as np
from jax import vmap
import numpy as onp
import cv2
from multi_rover_ergodic.erg_expl import ErgodicTrajectoryOpt
from multi_rover_ergodic.gaussian import gaussian
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from hillshade import get_shadow_map, get_shadow_map_stack, get_split_maps_idxs, map_time_series
from matplotlib.animation import FuncAnimation, FFMpegWriter
from info_distrib import random_info

def main(num_agents, map_size, time_args, info_map=None, pos=None, plot=False, shadows = None):
    if info_map is None:
        info_map = sample_map(map_size) #fix
    if pos is None:
        init_pos = sample_initpos(num_agents, map_size) #fix
        pos = [init_pos, init_pos]
        
    path_travelled = np.empty(shape=(num_agents, 2) + (0, )).tolist()

    traj_opt = ErgodicTrajectoryOpt(pos, info_map, num_agents, map_size, shadows, time_args)
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
        plt.show()

    return path_travelled

def main_sequential(num_agents, time_args, num_cells, map_size, info_map=None, shadow_idx_stack = None):
    
    path_travelled = np.empty(shape=(num_agents, 2) + (0, )).tolist()
    rows, cols = get_factors(num_cells)
    seq_shadows, startstop_list, world_offset, seq_pmap = sort_sequential(shadow_idx_stack, rows, cols, map_size, num_agents, num_cells, info_map)

    for i in range(num_cells):
        traj_opt = ErgodicTrajectoryOpt(startstop_list[i], seq_pmap[i], num_agents, map_size, seq_shadows[i], time_args)
        for k in range(100):
            traj_opt.solver.solve(max_iter=1000)
            sol = traj_opt.solver.get_solution()
            #clear_output(wait=False)

        for n in range(num_agents):
            path_travelled[n][0].append(sol['x'][:,n][:,0]+(world_offset[i][0] + map_size[1]/2))
            path_travelled[n][1].append(sol['x'][:,n][:,1]+(world_offset[i][1] + map_size[0]/2))

    np.save('path_data.npy', path_travelled)
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
    x_conv = np.round(pos_array[:,1]-size[1]/2)
    y_conv = np.round(pos_array[:,0]-size[0]/2)
    return np.vstack((x_conv, y_conv)).T

def get_factors(num):
    factors = []
    for i in range(1, num+1):
        if num%i==0:
            factors.append(i)
    factors = np.array(factors)
    if len(factors) %2 == 0:
        num_rows = factors[int(len(factors)/2 - 1)]
        num_cols = factors[int(len(factors)/2)]
    else:
        num_rows = int(np.median(factors))
        num_cols = int(np.median(factors))
    return num_rows, num_cols

def cell_paths(rows, cols, size, custom_start = [120,0]):
    start_pos = []
    end_pos = []
    world_offset = []

    for i in range(rows):
        for j in range(cols):
            if (i==0 and j==0):
                start_pos.append(custom_start)
                end_pos.append([size[0]//2, size[1]])
            elif i%2==0:
                if j==0:
                    start_pos.append([0, size[1]//2])
                    end_pos.append([size[0]//2, size[1]])
                elif j==(cols-1):
                    start_pos.append([size[0]//2, 0])
                    end_pos.append([size[0], size[1]//2])
                else:
                    start_pos.append([size[0]//2, 0])
                    end_pos.append([size[0]//2+1, size[1]])
            elif i%2==1:
                if j==(cols-1):
                    start_pos.append([0, size[1]//2])
                    end_pos.append([size[0]//2, 0])
                elif j==0:
                    start_pos.append([size[0]//2, size[1]])
                    end_pos.append([size[0], size[1]//2])
                else:
                    start_pos.append([size[0]//2, size[1]])
                    end_pos.append([size[0]//2+1, 0])
        
            world_offset.append([j*size[1], i*size[0]])

    return start_pos, end_pos, world_offset

def sort_sequential(shadow_idx_stack, rows, cols, size, num_agents, num_cells, pmap_list):
    start_pos, end_pos, world_offset = cell_paths(rows, cols, size)
    shadow_dims = np.shape(np.array(shadow_idx_stack))
    pmap_dims = np.shape(np.array(pmap_list))
    seq_shadow_idx_stack = []
    seq_start = []
    seq_end = []
    seq_world_offset = []
    seq_pmap = []

    for i in range(rows):
        if i%2==0:
            seq_shadow_idx_stack.append(shadow_idx_stack[i*cols:(i+1)*cols])
            seq_start.append(start_pos[i*cols:(i+1)*cols])
            seq_end.append(end_pos[i*cols:(i+1)*cols])
            seq_world_offset.append(world_offset[i*cols:(i+1)*cols])
            seq_pmap.append(pmap_list[i*cols:(i+1)*cols])
        else:
            seq_shadow_idx_stack.append(shadow_idx_stack[i*cols:(i+1)*cols][::-1])
            seq_start.append(start_pos[i*cols:(i+1)*cols][::-1])
            seq_end.append(end_pos[i*cols:(i+1)*cols][::-1])
            seq_world_offset.append(world_offset[i*cols:(i+1)*cols][::-1])
            seq_pmap.append(pmap_list[i*cols:(i+1)*cols][::-1])

    seq_start = np.reshape(np.array(seq_start), (num_cells, 2))
    seq_end = np.reshape(np.array(seq_end), (num_cells, 2))
    seq_world_offset = np.reshape(np.array(seq_world_offset), (num_cells, 2))
    seq_shadow_idx_stack = np.reshape(np.array(seq_shadow_idx_stack), shadow_dims)
    seq_pmap = np.reshape(np.array(seq_pmap), pmap_dims)
    
    startstop_list = []
    for k in range(len(seq_start)):
        start = convert_pos(np.tile(seq_start[k], (num_agents, 1)), size)
        stop = convert_pos(np.tile(seq_end[k], (num_agents, 1)), size)
        startstop_list.append([start, stop])
    
    return seq_shadow_idx_stack, startstop_list, seq_world_offset, seq_pmap

def animate_plot(path_travelled, num_agents, time_horizon, pmap, shadow_map_stack, num_cells=1):
    
    fps = 20
    size = pmap.shape[0]
    cmap = get_colormap(num_agents+1)
    pos_x = []
    pos_y = []
    for i in range(num_agents):
        pos_x.append(np.array(path_travelled[i][0]).flatten()) 
        pos_y.append(np.array(path_travelled[i][1]).flatten())

    fig, ax = plt.subplots()
    img = ax.imshow(shadow_map_stack[0], cmap='Greys_r', origin='upper', animated = True)
    '''
    num_ticks = len(ax.get_xticks())
    tick_labels = np.linspace(0, 16, num_ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    '''

    plt.xlabel('km')
    plt.ylabel('km')
    #plt.title('Timescale x' + str(total_time/(time_horizon/fps)))
    
    overlay = ax.imshow(pmap, origin='upper', alpha = .5, animated = True)
    cbar = plt.colorbar(overlay)
    cbar.ax.set_title('Information Density')

    for i in range(num_agents):
        line = [[pos_x[i][0], pos_x[i][1]], [pos_y[i][0], pos_y[i][1]]]
        traj, = ax.plot(line[0], line[1], c=cmap(i))

    
    max_frames = len(shadow_map_stack)
    def updatefig(frame, img, traj, ax):
        img.set_array(shadow_map_stack[frame])
        overlay.set_array(pmap)
        for i in range(num_agents):
            line = [[pos_x[i][frame], pos_x[i][frame+1]], [pos_y[i][frame], pos_y[i][frame+1]]]
            traj, = ax.plot(line[0],line[1], c=cmap(i))
        return img, traj

    ani = FuncAnimation(fig, updatefig, frames=time_horizon*num_cells, fargs=(img, traj, ax), blit=True)
    FFwriter = FFMpegWriter(fps=fps, codec='libx264', bitrate=1800)
    ani.save('shadow_avoidance.mp4', writer=FFwriter)


def split(info_map, num_cells):
    num_rows, num_cols = get_factors(num_cells)
    row = info_map.shape[0]
    col = info_map.shape[1]
    
    bounds = []
    y_bound = int(row/num_rows)
    x_bound = int(col/num_cols)
    for i in range(num_rows):
        y_min = int(i*y_bound)
        y_max = int((i+1)*y_bound)
        for j in range(num_cols):
            bounds.append([[int(j*x_bound), int((j+1)*x_bound)], [y_min, y_max]])
    bounds = np.array(bounds)
        
    split_maps = []
    for bound in bounds:
        cell = info_map[bound[1,0]:bound[1,1], bound[0,0]:bound[0,1]]
        split_maps.append(cell)
    
    return split_maps

#######################################################

dem_path = "DEMs/Site01_final_adj_5mpp_surf.tif"
time_args = {
    'dt': 100,
    'start_time': 0,
    'end_time': 10000,
    'time_horizon': 100
}
num_cells = 4

split_shadow_stack, split_idx_stack, original_shadow_stack = get_split_maps_idxs(dem_path, time_args, num_cells) #split shadows sorted by row then column
split_shadow_stack, split_idx_stack = map_time_series(split_shadow_stack, split_idx_stack, num_cells, time_args['time_horizon']*num_cells)

'''
shadow_map_stack = split_shadow_stack[0]
shadow_idx_stack = split_idx_stack[0]

shadow_map = shadow_map_stack[0] #TODO: update info map to change over time
size = [np.shape(shadow_map)[0], np.shape(shadow_map)[1]]
start_pos = np.array([[30, 70], [40, 70], [50, 70]])
end_pos  = np.array([[70,40], [70, 40], [70, 40]])
init_pos = convert_pos(start_pos, size)
final_pos = convert_pos(end_pos, size)
startstop = [init_pos, final_pos]
'''

original_size =  [np.shape(original_shadow_stack[0])[0], np.shape(original_shadow_stack[0])[1]]
pmap = random_info(original_size)
pmap_list = split(pmap, num_cells)

size = [np.shape(split_shadow_stack[0][0])[0], np.shape(split_shadow_stack[0][1])[1]]
main_sequential(3, time_args, num_cells, map_size = size, info_map=pmap_list, shadow_idx_stack=split_idx_stack)
path_travelled = np.load('path_data.npy')
animate_plot(path_travelled, 3, time_args['time_horizon'], pmap, original_shadow_stack, num_cells)

# for testing:
#pmap = np.ones((size[0], size[1]))
#shadow_idx_stack = np.ones((100, 1, 2))*50

'''
main(num_agents = 3, map_size = size, time_args = time_args, pos = startstop, info_map = pmap, shadows = shadow_idx_stack)
path_travelled = np.load('path_data.npy')
animate_plot(path_travelled, 3, time_args['time_horizon'], pmap, shadow_map_stack)
'''

#TODO: make scaling depend on map size
#TODO: all agents still aren't showing in plotting. ask chatgpt