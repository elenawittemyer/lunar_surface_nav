import jax.numpy as np
import numpy as onp
from multi_rover_ergodic.gaussian import gaussian
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def convert_pos(pos_array, size):
    return np.round(pos_array-size/2)

def crater_pmap(crater_pos, size, shadow_map):
    pmap = np.ones((size, size))
    pmap = pmap/(size*size)
    for i in range(0, len(crater_pos)):
        pmap += .01*gaussian(size, crater_pos[i][0], crater_pos[i][1], 10)
    return pmap

def random_pmap(size):
    points = onp.random.rand(size, size)
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='blue',
                      line_width=2, line_alpha=0.6, point_size=8)
    plt.show()

random_pmap(100)