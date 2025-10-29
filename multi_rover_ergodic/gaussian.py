
# Importing Numpy package
import jax.numpy as np
import random
import matplotlib.pyplot as plt
from functools import partial
from jax import jit

def gaussian_filter(kernel_size, sigma=0.3, muu=0):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
 
    # lower normal part of gaussian
    normal = 1.0/(2.0 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal

    return gauss

def random_gaussian(size, num_peaks):

    # Initialize x and y
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    # Initialize the gaussian filter
    gaussian = np.zeros((size, size))

    # Randomly generate peaks for this gaussian
    for i in range(0, num_peaks):
        # Center point
        x0 = random.randint(0,size)
        y0 = random.randint(0,x0)

        # Effective "radius" of the peak
        radius = random.randint(size/5, int(size/1.5))

        gaussian += np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / radius**2)

    return gaussian
 
def gaussian(size, x0, y0, radius):

    # Initialize x and y
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    # Initialize the gaussian filter
    gaussian = np.zeros((size, size))

    gaussian += np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / radius**2)
    
    return gaussian

def gaussian_1d(spread, x0, radius):
    x = np.arange(0, spread, 1, float)
    gaussian = np.zeros(spread)
    gaussian += np.exp(-4*np.log(2)*(x-x0)**2 / radius**2)
    return gaussian
 
def gaussian_measurement(size, x0, y0, spread):
    rad = size*spread
    peak_reduction = -1*gaussian(size, x0, y0, rad)
    return peak_reduction

'''
kernel_size=200
gaussian = gaussian_filter(kernel_size)
big_gaussian1 = np.hstack((np.zeros((200,200)), np.zeros((200,200))))
big_gaussian2 = np.hstack((np.zeros((200,100)), gaussian, np.zeros((200,100))))
big_gaussian = np.vstack((big_gaussian1, big_gaussian2))
big_gaussian = np.max(big_gaussian) - big_gaussian
big_gaussian1 = np.hstack((np.ones((200,200)), np.ones((200,200))))
big_gaussian2 = np.hstack((np.ones((200,200))*0.01, np.ones((200,200))))
big_gaussian = np.vstack((big_gaussian1, big_gaussian2))

for i in range(0, 400):
    for j in range(0, 400):
        print(big_gaussian[i][j], end=" ")
    print("")

plt.imshow(big_gaussian.T, origin="lower")
plt.show()

# size = 400
# rand_gaussian = random_gaussian(size, 20)
# rand_gaussian = np.max(rand_gaussian) - rand_gaussian
# rand_gaussian = rand_gaussian / np.sum(rand_gaussian)

# for i in range(0, size):
#     for j in range(0, size):
#         print(rand_gaussian[i][j], end=" ")
#     print("")

# fig = plt.figure()
# plt.imshow(rand_gaussian.T, origin="lower")

# plt.show()
'''

# pixels, x loc of center point, y loc of center point, radius of gaussian
# rand_gaussian1 = gaussian(size, 420, 100, 200)
# rand_gaussian2 = gaussian(size, 420, 150, 100)

# rand_gaussian3 = gaussian(size, 200, 180, 120)

# rand_gaussian4 = gaussian(size, 100, 350, 100)
# rand_gaussian5 = gaussian(size, 150, 400, 75)
# rand_gaussian6 = gaussian(size, 150, 350, 120)
# rand_gaussian7 = gaussian(size, 75, 425, 75)
# rand_gaussian8 = gaussian(size, 50, 400, 50)
# rand_gaussian9 = gaussian(size, 50, 375, 50)

#rand_gaussian1 = gaussian(size, 420, 125, 220)
#rand_gaussian2 = gaussian(size, 200, 180, 130)
#rand_gaussian3 = gaussian(size, 96, 383, 250)

#rand_gaussian = rand_gaussian1 + rand_gaussian2 + rand_gaussian3 +rand_gaussian4 + rand_gaussian5 + rand_gaussian6 + rand_gaussian7 + rand_gaussian8 + rand_gaussian9
#print(np.max(rand_gaussian))
#print(np.min(rand_gaussian))
# rand_gaussian = rand_gaussian - np.min(rand_gaussian)
# rand_gaussian = rand_gaussian / np.sum(rand_gaussian)

# for i in range(0, size):
#     for j in range(0, size):
#         print(rand_gaussian[i][j], end=" ")
#     print("")

# np.save("../cost_maps/test.npy", 1.0 - rand_gaussian)
#fig = plt.figure()
#plt.imshow(rand_gaussian2.T, origin="lower")
#plt.colorbar()
# plt.savefig("../cost_maps/test.png", format="png")

#plt.show()
