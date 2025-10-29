import jax.numpy as np
import numpy as onp
from jax import vmap, jit
from .gaussian import gaussian_1d, gaussian
import matplotlib.pyplot as plt

class TargetDistribution(object):
    def __init__(self, pmap, size) -> None:
        self.n = 2
        self.domain = np.meshgrid(
            *[np.linspace(0.01, 0.99, size)]*self.n 
        )
        pmap=pmap.reshape((1, size**2))
        pmap=pmap[0]
        self._s = np.stack([X.ravel() for X in self.domain]).T
        self.evals = (
            self.p(pmap) , self._s
        )

    def p(self, map):
        return map

    def update(self):
        pass