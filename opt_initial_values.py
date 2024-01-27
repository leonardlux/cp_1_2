import numpy as np
import config as c

# generate inital value
def inital_values_dirac():
    # this basic dirac put the total mass in one field
    u_0 = np.zeros(c.n_x)
    i_0 = int((c.x_0-c.x_min)/c.dx)
    u_0[i_0] = c.initial_commulative_mass / c.dx
    return u_0

def inital_values_gauss(mu=c.x_0,sigma=(c.x_max-c.x_min)/20):
    u_0 = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (c.x_g - mu)**2 / (2 * sigma**2)) *c.initial_commulative_mass
    # scale to have correct inital mass in the system, even if part of the distribution is outside
    return u_0

def inital_values_uniform():
    u_0 = np.ones(c.n_x, dtype=c.dtype) * c.initial_commulative_mass / c.n_x
    return u_0

