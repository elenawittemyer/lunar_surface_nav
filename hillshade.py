import matplotlib.pyplot as plt
import numpy as np
import rasterio
import earthpy.spatial as es
import cv2
import os
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LightSource
from matplotlib import cbook, cm

lunar_day_min = 42524 #minutes in a lunar day

def get_shadow_map(dem_path, scaling = 10, plotting = False):
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        geometry = [src.bounds]
        hillshade = es.hillshade(dem)
        
        shadow_idx = np.where(hillshade>210)
        shadow_map = 100*np.ones(np.shape(hillshade))
        shadow_map[shadow_idx] = 0
        resized_x = shadow_map.shape[1] // scaling
        resized_y = shadow_map.shape[0] // scaling
        resized_shadow_map = cv2.resize(shadow_map, (resized_x, resized_y), interpolation=cv2.INTER_AREA)
        
        if plotting == True:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,6))
            ax1.imshow(hillshade, cmap='Greys', origin='upper')
            ax2.imshow(shadow_map, cmap='Greys_r', origin='upper')
            ax3.imshow(resized_shadow_map, cmap='Greys_r', origin='upper')
            ax1.set_title('Site 01 - Hillshade')
            ax2.set_title('Site 01 - Shadows')
            ax3.set_title('Site 01 - Reduced Resolution Shadows')
        plt.show()

        return resized_shadow_map

def get_shadow_map_stack(dem_path, site_name, start_time=0, end_time=60, dt=5, scaling=10, plotting=False):
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        shadow_map_stack = []
        for i in range((end_time-start_time)//dt):
            '''
            azimuth = (start_time+i*dt) / lunar_day_min * 360
            hillshade = es.hillshade(dem, azimuth, altitude=5)
            
            shadow_idx = np.where(hillshade>210)
            shadow_map = 100*np.ones(np.shape(hillshade))
            shadow_map[shadow_idx] = 0
            '''
            
            sm_save_file = 'shadow_maps/' + site_name + '/shadows_' + str(start_time+i*dt) +'.npy'
            if os.path.isfile(sm_save_file) == False:
                azimuth = ((start_time+i*dt) / lunar_day_min * 360)%360
                hillshade = es.hillshade(dem, azimuth, altitude=5)
                
                shadow_idx = np.where(hillshade>210)
                shadow_map = 100*np.ones(np.shape(hillshade))
                shadow_map[shadow_idx] = 0
                
                resized_x = shadow_map.shape[1] // scaling
                resized_y = shadow_map.shape[0] // scaling
                resized_shadow_map = cv2.resize(shadow_map, (resized_x, resized_y), interpolation=cv2.INTER_AREA)
                np.save(sm_save_file, resized_shadow_map)
                
            else:
                resized_shadow_map = np.load(sm_save_file)
            
            #resized_shadow_map = shadow_map[bounds[0, 0]:bounds[0, 1], bounds[1, 0]:bounds[1, 1]]
            shadow_map_stack.append(resized_shadow_map)

            
        if plotting == True:
            fig, ax = plt.subplots()
            im = ax.imshow(shadow_map_stack[0], cmap = 'Greys_r', origin='upper', animated=True)
            def update(i):
                im.set_array(shadow_map_stack[i+1])
                return im,
            
            ani = FuncAnimation(fig, update, frames=(end_time-start_time)//dt - 1, blit=True)
            plt.show()

        return shadow_map_stack
    

def get_factors(num):
    factors = []
    for i in range(1, num+1):
        if num%i==0:
            factors.append(i)
    return factors

def map_splitter(shadow_map_stack, num_cells=4, bounds=None):
    split_shadow_stack = []
    for shadow_map in shadow_map_stack:
        if bounds is None:
            if num_cells %2 == 0:
                factors = get_factors(num_cells)
                if len(factors) %2 == 0:
                    num_rows = factors[int(len(factors)/2 - 1)]
                    num_cols = factors[int(len(factors)/2)]
                else:
                    num_rows = int(np.median(factors))
                    num_cols = int(np.median(factors))
            else:
                raise Exception('Map must be split into an even number of cells')
            row = shadow_map.shape[0]
            col = shadow_map.shape[1]
            bounds = []

            for i in range(0, num_rows):
                y_min = int((row/num_rows)*i)
                y_max = int((row/num_rows)*(i+1))
                for j in range(0, num_cols):
                    x_min = int((col/num_cols)*j)
                    x_max = int((col/num_cols)*(j+1))
                    
                    bounds.append([[x_min, x_max], [y_min, y_max]])
            bounds = np.array(bounds)
        
        split_maps = []
        for bound in bounds:
            cell = shadow_map[bound[1,0]:bound[1,1], bound[0,0]:bound[0,1]]
            split_maps.append(cell)
        
        split_shadow_stack.append(split_maps)
    
    return split_shadow_stack

def convert_idx(idx_array, size):
    row_conv = np.round(idx_array[:,0]-size[0]/2)
    col_conv = np.round(idx_array[:,1]-size[1]/2)
    return np.vstack((row_conv, col_conv)).T #TODO: fix this
        
def get_split_maps_idxs(path, time_args, num_maps):
    
    time_horizon = time_args['time_horizon']
    start_time = time_args['start_time']
    end_time = time_args['end_time']
    dt = time_args['dt']

    shadow_map_stack = get_shadow_map_stack(path, 'Site01', start_time, end_time, dt)
    split_stack = map_splitter(shadow_map_stack, num_maps)
    
    if ((end_time-start_time)//dt)!=time_horizon:
        raise Exception('Time horizon and number of time steps do not match.')
    
    shadows_idx_stack = []
    for i in range(len(split_stack)):
        for j in range(num_maps):
            shadow_map = split_stack[i][j]
            scale = 10/num_maps
            resized_x = int(shadow_map.shape[1] // scale)
            resized_y = int(shadow_map.shape[0] // scale)
            resized_shadow_map = cv2.resize(shadow_map, (resized_x, resized_y), interpolation=cv2.INTER_AREA)
            shadow_idx = np.where(resized_shadow_map<40)
            shadow_idx_array = scale*np.array([shadow_idx[1], shadow_idx[0]]).T
            shadow_idx_array = convert_idx(shadow_idx_array, [np.shape(shadow_map)[0], np.shape(shadow_map)[1]]) #rows vs cols?
            shadows_idx_stack.append(shadow_idx_array)
    
    def padding(map_idx, max_len, map_size):
        current_len = map_idx.shape[0]
        padded_vals_x = map_size[1]*np.ones(max_len-current_len)
        padded_vals_y = map_size[0]*np.ones(max_len-current_len)
        padded_vals = np.vstack((padded_vals_x, padded_vals_y)).T
        #padded_vals = map_size[0]*np.ones((max_len-current_len, 2))
        return np.vstack((map_idx, padded_vals))

    max_len = max(arr.shape[0] for arr in shadows_idx_stack)
    for j in range(num_maps):
        for i in range(len(split_stack)): #this should be len(shadows_idx_stack). figure out how to correct with i, j
            idx = (j*100)+i
            map_size =  np.shape(split_stack[i][j])
            if len(shadows_idx_stack[idx])<max_len:
                shadows_idx_stack[idx] =  padding(shadows_idx_stack[idx], max_len, map_size)
    shadows_idx_stack = np.reshape(np.array(shadows_idx_stack), (len(split_stack), num_maps, max_len, 2))
    
    split_stack_list = []
    idx_stack_list = []
    for i in range(num_maps):
        idx_stack_list.append(shadows_idx_stack[:,i])
    
    for i in range(num_maps):
        cells = []
        for j in range(len(split_stack)):
            cell = split_stack[j][i]
            cells.append(cell)
        cells = np.array(cells)
        split_stack_list.append(cells)

    return split_stack_list, idx_stack_list

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
        shadow_idx_array = convert_pos(shadow_idx_array, [np.shape(shadow_map)[0], np.shape(shadow_map)[1]])
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


bounds = np.array([[[0, 10], [0, 10]], [[10, 20], [10, 20]]]) #[[x_min, x_max], [y_min, y_max]]
dem_path = "DEMs/Site01_final_adj_5mpp_surf.tif"
time_args = {
    'dt': 100,
    'start_time': 0,
    'end_time': 10000,
    'time_horizon': 100
}

#shadow_map_stack = get_shadow_map_stack(dem_path, 'Site01')
#split_stack = map_splitter(shadow_map_stack)
split_stack, split_idxs = get_split_maps_idxs(dem_path, time_args, 4)



'''
path = "DEMs/Site01_final_adj_5mpp_surf.tif"
with rasterio.open(path) as src:
    dem = src.read(1)
    geometry = src.bounds
    cols, rows = dem.shape
    x = np.linspace(geometry[0], geometry[2], cols)
    y = np.linspace(geometry[1], geometry[3], rows)
    x, y = np.meshgrid(x,y)

    region = np.s_[0:1000, 0:1000]
    x = x[region] 
    y = y[region]
    z = dem[region]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ls = LightSource(270, 45)
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                        linewidth=0, antialiased=False, shade=False)
    plt.show()
'''