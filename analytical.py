import config as c
import numpy as np
from time import time

def timer(p_time, n_string=""):
    print(n_string,time()-p_time)




# Analytical solutions without boundary conditions

def __analytical_solution_unwrapped(t,x):
    u = c.initial_commulative_mass/np.sqrt(4*np.pi*c.D_c*t) * np.exp(-1*(x-c.x_0)**2 /(4*c.D_c*t))
    return u

def analytical_solution(t,x,renomarlize = False):
    #wrapped because i am to lazy for np.frompyfunc
    U = np.zeros((len(t),len(x)))
    for i, t_ in enumerate(t[1:]):
        U[i] = __analytical_solution_unwrapped(t_,x)

    if renomarlize:
        # renormalization:
        U = U/np.sum(U[0])*c.initial_commulative_mass
    return U 


# Analytical solutions with boundary conditions

L = c.x_max-c.x_min

def mo(a,b):
    return np.multiply.outer(a,b)

def v_neumann(x,n):
    # refelective
    v = np.sqrt(2/L) * np.cos(np.pi*mo(x,n)/L)
    v[:,0] = np.sqrt(1/L) * np.ones(len(x))
    return v

def v_dirichlet(x,n):
    # absorbing
    v = np.sqrt(2/L) * np.sin(np.pi*mo(x,n)/L)
    v[:,0] = np.zeros(len(x))
    return v
    
# this sadly uses about 4.7 GB of RAM, with n_max = 1e3 and same points as sim
# Therefore I needed to implement a mask to limit the t_values

def analytical_solution_boundary(t,x,cond: str,n_max = 1e2):
    # choose correct function
    if cond.lower() == "dirichlet" or cond.lower() == "d":
        v = v_dirichlet
    elif cond.lower() == "neumann" or cond.lower() == "n":
        v = v_neumann
    else:
        print(f"analytical_solution_boundary: {cond=}")
        raise NameError

    n = np.arange(0,n_max)
    # idea was to expand the calculation to a 3d array [t,x,n]
    # then sum over the last index to get all values
    v_term = mo(np.ones(len(t)),v(x,n) * v(np.array([c.x_0]),n))
    exponent_term = np.exp(-(np.pi/L)**2*c.D_c*mo(mo(t,np.ones(len(x))),n**2))
    U = c.initial_commulative_mass * np.sum(exponent_term * v_term ,axis=2)

    return U


