import jax.numpy as np

class SingleIntegrator(object):
    def __init__(self, N) -> None:        # N = the number of drones
        self.dt = 1   # timestep
        self.n = 2      # dimensionality of the states (x, y, z) would be 3
        self.m = 2      # dimensionality of the controls (vx, vy, vz) would be 3
        self.N = N
        B = np.array([
                [1.,0.],
                [0.,1.],
            ])
        def _f(x, u):
            return x + self.dt*B@u
        def f(x1, u1):
            # assumes x1 a Nxn, u1 a Nxm dim
            x2 = []
            for i in range(self.N):
                x2.append(_f(x1[i,:], u1[i,:]))
            return np.stack(x2)
        self._f = _f
        self.f = f

def _f_test(x, u):
    dt = 1
    B = np.array([[1.,0.],
                  [0.,1.]],)
    return x + dt*B@u

def f_test(x, u, N):
    x2 = []
    for i in range(N):
        x2.append(_f_test(x[i,:], u[i,:]))
    return np.stack(x2)