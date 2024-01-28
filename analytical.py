import config as c
import numpy as np
import scipy as sc
from time import time

def timer(p_time, n_string=""):
    print(n_string,time()-p_time)




# Analytical solutions without boundary conditions

def __analytical_solution_unwrapped(t,x):
    u = c.initial_commulative_mass/np.sqrt(4*np.pi*c.D_c*t) * np.exp(-1*(x-c.x_0)**2 /(4*c.D_c*t))
    return u

def analytical_solution(t,x):
    #wrapped because i am to lazy for np.frompyfunc
    U = np.zeros((len(t),len(x)))
    for i, t_ in enumerate(t[1:]):
        U[i] = __analytical_solution_unwrapped(t_,x)
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


# Analytical solution for the stepped diffusivity
# I split the function up into smaller easier to handle funtions, if some parts appear more than once.
def calc_erf(t,d):
    return sc.special.erf(c.x_0/np.sqrt(4*t*d))

def calc_a_exp(t):
    d_p = c.D_plus
    d_m = c.D_minus
    return np.exp((d_p-d_m)*c.x_0**2/(4*d_p*d_m*t))

def calc_A_plus(t):
    d_p = c.D_plus
    d_m = c.D_minus
    a_plus = 2 * ( 1 
                  + calc_erf(t,d_p) 
                  + ( np.sqrt(d_m/d_p) 
                     * calc_a_exp(t)
                     * calc_erf(t,d_m)
                     )
                  )**-1
    return a_plus

def calc_A_minus(t):
    d_p = c.D_plus
    d_m = c.D_minus
    a_minus = (calc_A_plus(t) * 
               np.sqrt(d_m/d_p) * calc_a_exp(t)
               )
    return a_minus

# Exponential funtion of the base function u(x,t)/u~
def calc_base_exp(x,t,d):
    return np.exp(-1* (x-c.x_0)**2/(4*d*t))

# Prefactor of the base function u(x,t)/u~
def calc_base_pre(t,func_calc_A,d):
    return func_calc_A(t)/np.sqrt(4*np.pi*d*t)

def calc_total_plus(x,t):
    return calc_base_pre(t,calc_A_plus,c.D_plus) * calc_base_exp(x,t,c.D_plus)

def calc_total_minus(x,t):
    return calc_base_pre(t,calc_A_minus,c.D_minus) * calc_base_exp(x,t,c.D_minus)

def analytical_solution_stepped(x,t,mult_with_u_0=True):
    # just iterate through x (significantly smaller than t)
    U_inverted = [0]*len(x) # ugly but simple way to build an list with the correct length
    for i, x_i in enumerate(x):
        if   x_i >= 0:
            U_inverted[i] = calc_total_plus(x_i,t)
        elif x_i < 0:
            U_inverted[i] = calc_total_minus(x_i,t)
    U = np.swapaxes(np.array(U_inverted),0,1)
    if mult_with_u_0:
        U = U * c.initial_commulative_mass
    return U 