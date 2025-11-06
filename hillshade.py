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

def get_shadow_map_stack(dem_path, site_name, bounds, start_time=0, end_time=60, dt=5, scaling=10, plotting=False):
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
                azimuth = (start_time+i*dt) / lunar_day_min * 360
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
    

#get_shadow_map_stack(path, 'Site01', start_time=0, dt=200, end_time=20000, plotting=True)