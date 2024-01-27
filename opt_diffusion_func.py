import numpy as np
import config as c

# different diffusivity Functions
def diffusivity_const(x):
    return np.ones(len(x)) * c.D_c

def diffusivity_step(x):
    return np.piecewise(x, [x < 0, x >= 0], [c.D_minus,c.D_plus])
