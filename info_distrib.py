import numpy as np
from multi_rover_ergodic.gaussian import gaussian
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex

def convert_pos(pos_array, size):
    return np.round(pos_array-size/2)

def crater_pmap(crater_pos, size):
    pmap = np.ones((size, size))
    pmap = pmap/(size*size)
    for i in range(0, len(crater_pos)):
        pmap += .01*gaussian(size, crater_pos[i][0], crater_pos[i][1], 10)
    return pmap

def random_info(size, plotting=False):
    seed = np.random.randint(0, 10000)
    noise_generator = OpenSimplex(seed)

    width = size
    height = size
    scale = 80.0  # Controls the "zoom" level of the noise
    octaves = 1   # Number of noise layers
    persistence = .5 # How much each octave contributes
    lacunarity = 2.0 # Frequency multiplier for each octave

    elevation_map = np.zeros((height, width))

    # Generate elevation data using Perlin noise
    for y in range(height):
        for x in range(width):
            amplitude = 1.0
            frequency = 1.0
            noise_value = 0
            for i in range(octaves):
                # Sample noise at current coordinates, scaled by frequency
                noise_value += noise_generator.noise2(x / scale * frequency, y / scale * frequency) * amplitude
                amplitude *= persistence
                frequency *= lacunarity
            elevation_map[y, x] = noise_value

    elevation_map = (elevation_map - np.min(elevation_map)) / (np.max(elevation_map) - np.min(elevation_map))

    if plotting == True:
        plt.imshow(elevation_map, origin='upper')
        plt.colorbar(label='information')
        plt.show()
    return elevation_map