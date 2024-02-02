import numpy as np
import config as c

# different diffusivity Functions
def diffusivity_const(x):
    return np.ones(len(x)) * c.D_c

def diffusivity_step(x):
    return np.piecewise(x, [x < c.x_step, x >= c.x_step], [c.D_minus,c.D_plus])

def diffusivity_special(x):
    # approximation of picewiese function
    a = 100
    return 1/(1+np.exp(-a*x))* ( c.D_plus - c.D_minus ) + c.D_minus

# def diffusivity_special(x):
#     return (1 + x**3) * 2 + 0.2
