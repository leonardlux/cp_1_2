
import config as c
import numpy as np

"""
# Here live the function, which generate the matricies 
"""

def diags_gen_simple_version(*args):
    """
    We do not need the d_func, and therefore replace it with *args
    assumes D(x) = const.
    """
    # generate diagonals
    a_main = np.ones(c.n_x  , dtype=c.dtype) * ( 1 + 2 * c.alpha)
    a_sup  = np.ones(c.n_x , dtype=c.dtype) * (-1 * c.alpha)
    a_sub  = np.ones(c.n_x , dtype=c.dtype) * (-1 * c.alpha)
    a_diags = (a_main,a_sup,a_sub)
    
    b_main  = np.ones(c.n_x  , dtype=c.dtype) * ( 1 - 2 * c.alpha)
    b_sup   =  np.ones(c.n_x , dtype=c.dtype) * ( 1 * c.alpha)
    b_sub   =  np.ones(c.n_x , dtype=c.dtype) * ( 1 * c.alpha)
    b_diags = (b_main,b_sup,b_sub)

    return a_diags, b_diags


def diags_gen_general_version(d_func):
    """
    general version does not assume D(x) to e constand
    """
    # functions to calulate the alphas
    def f_alpha_1(x):
        """
        x: 
            array of x values for which the D_values are to be calculated
        --------
        alpha_1 = Delta t * D_i / (2 * ([Delta x]**2))
        """
        alpha_1 = (d_func(x)*c.dt)/(2*c.dx**2)
        return alpha_1
    
    def f_alpha_2(x):
        """
        x: 
            array of x values for which the D_values are to be calculated
        --------
        alpha_1 = Delta t / (8 * ([Delta x]**2)) * ( D_(i+1) - D_(i-1))
        with D_i+1: D(x+d_x)
        and D_i-1: D(x-d_x)
        """ 
        alpha_2 = c.dt / (8*c.dx**2) * (d_func(x+c.dx)- d_func(x-c.dx))
        return alpha_2

    # generate x_values for the different diagonals
    # sup and sub diags, will be reduced to correct size by boudary conditions 
    
    # generate diags
    a_main  = 1 + 2 * f_alpha_1(c.x_g)
    a_sup   = ( -1 * f_alpha_1(c.x_g) - f_alpha_2(c.x_g) )
    a_sub   = ( -1 * f_alpha_1(c.x_g) + f_alpha_2(c.x_g) )
    a_diags = (a_main,a_sup,a_sub)
    

    b_main  = 1 - 2 * f_alpha_1(c.x_g)
    b_sup   = ( f_alpha_1(c.x_g) + f_alpha_2(c.x_g) )
    b_sub   = ( f_alpha_1(c.x_g) - f_alpha_2(c.x_g) )
    b_diags = (b_main,b_sup,b_sub)

    return a_diags, b_diags