import config as c
import numpy as np

from scipy.sparse import diags   
from numba import jit


import opt_initial_values     as opt_sv
import opt_diags_gen    as opt_dg
import opt_diffusion_func     as opt_df
import opt_boundary_conditions  as opt_bc
from plots import plot_mass, plot_diff_times


def build_matrix_from_diag(diags_list):
    # unpack list
    main_diag, sup_diag, sub_diag, = diags_list
    # build a banded matrix using the different diagonals
    M = diags((sub_diag, main_diag, sup_diag), offsets = (-1, 0, 1), dtype=c.dtype)# .toarray()
    return M

def gen_matricies(diags_gen,d_func, boundary_condition):
    # generate diags
    a_diags, b_diags = diags_gen(d_func)
    # apply boundary conditons
    a_diags = boundary_condition(a_diags)
    b_diags = boundary_condition(b_diags)
    # build Matrix
    A = build_matrix_from_diag(a_diags)
    B = build_matrix_from_diag(b_diags)
    return A,B


#taken from github of this course, after scipy did not work :( 
@jit(nopython = True)
def tdma_solver(a, b, c, d):
    # Solves Ax = d,
    # where layout of matrix A is
    # b1 c1 ......... 0
    # a2 b2 c2 ........
    # .. a3 b3 c3 .....
    # .................
    # .............. cN-1
    # 0 ..........aN bN
    # Note index offset of a
    N = len(d)
    # Make to extra arrays to avoid overwriting input
    c_ = np.zeros(N-1)
    d_ = np.zeros(N)
    x  = np.zeros(N)
    c_[0] = c[0]/b[0]
    d_[0] = d[0]/b[0]
    for i in range(1, N-1):
        q = (b[i] - a[i-1]*c_[i-1])
        c_[i] = c[i]/q
        d_[i] = (d[i] - a[i-1]*d_[i-1])/q
    d_[N-1] = (d[N-1] - a[N-2]*d_[N-2])/(b[N-1] - a[N-2]*c_[N-2])
    x[-1] = d_[-1]
    for i in range(N-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]
    return x

def tdma(A, b):
    # Solves Ax = b to find x
    # This is a wrapper function, which unpacks
    # A from a sparse array structure into separate diagonals,
    # and passes them to the numba-compiled solver defined above.
    x = tdma_solver(A.diagonal(-1), A.diagonal(0), A.diagonal(1), b)
    return x


# solve_simulation
def solve_simulation(parameter: dict):
    # unpack parameter
    starting_distr      = parameter["starting_distr"]       # u(x,t=0) as a array
    # for matrix
    diags_gen           = parameter["diags_gen"]            # function which generates the diagonals  
    d_func              = parameter["d_func"]               # D(x) as a function
    boundary_condition  = parameter["boundary_condition"]   # Function that changes the matrix

    # initalize U
    U  = np.zeros((c.n_t,c.n_x), dtype=c.dtype)
    U[0] = starting_distr()

    # generate Matricies
    A, B = gen_matricies(diags_gen,d_func, boundary_condition)
    
    for n in range(0, c.n_t-1):
        # Calculate matrix-vector product: B * u^n = _u (right hand side)
        _u = B.dot(U[n,:])
        # Then, solve equation A * u^(n+1) = _u
        #U[n+1,:] = solve_banded((lower_bands,upper_bands), A_b, _u,)
        # the scipy version failed, therefore I used the solver provided in the github
        U[n+1,:] = tdma(A,_u)
    
    return U


# Object:

class Simulation:
    def __init__(self,
                distribution        : str  = "dirac",
                diags_gen           : str  = "simpel",
                d_func              : str  = "constant",
                boundary_condition  : str  = "dirichlet", 
                overwrite           : bool = False,
                data_dict           : dict = {},
                ) -> None:
        """
        This object always has self.title and self.U 
        """
        if overwrite:
            self.U = data_dict["U"]
            self.title = data_dict["title"]
        else:
            self.sim_para = {}

            # Starting distribution
            if   distribution.lower() in "gauss":
                self.sim_para["starting_distr"] = opt_sv.inital_values_gauss
                self.dist_name = "Gaussian"
            elif distribution.lower() in "dirac":
                self.sim_para["starting_distr"] = opt_sv.inital_values_dirac
                self.dist_name = "Dirac-Delta"
            else: raise ValueError

            # Diagonal Generation Function
            if   diags_gen in "general":
                self.sim_para["diags_gen"] = opt_dg.diags_gen_general_version
            elif diags_gen in "simpel":
                self.sim_para["diags_gen"] = opt_dg.diags_gen_simple_version
            else: raise ValueError

            # D(x) Function
            if   d_func in "constant":
                self.sim_para["d_func"] = opt_df.diffusivity_const
            elif d_func in "step":
                self.sim_para["d_func"] = opt_df.diffusivity_step
            else: raise ValueError

            # Boundary Condtions
            if   boundary_condition.lower() in "neumann" :
                self.sim_para["boundary_condition"] = opt_bc.boundary_conditions_neumann
                self.bc_name = "Neumann"
            elif boundary_condition.lower() in "dirichlet" :
                self.sim_para["boundary_condition"]  = opt_bc.boundary_conditions_dirichlet
                self.bc_name = "Dirichlet"
            else: raise ValueError
            
            # Solve Simulation
            self.U = solve_simulation(self.sim_para)
            self.title = f"BC: {self.bc_name}, Distr.: {self.dist_name}"
        pass

    
    def plot_mass(self):
        plot_mass(self.U,self.title)
    
    def plot_diff_times(self):
        # not usefull right now
        # needs rework
        plot_diff_times(self.U,self.title)        