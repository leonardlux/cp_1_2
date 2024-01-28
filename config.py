import numpy as np
# Set parameters

# dtype 
dtype = np.float64



# number points used in discretized space and time
# space
n_x = 200 +1 
# +1 one because we want to include the 0 and still have a nice spacing

x_min = -1
x_max =  1
x_0   = 0.5

# globaly used steps for x
x_g, dx = np.linspace(x_min,x_max,n_x,retstep=True,dtype=dtype)

#TODO explain the -1 and 201 better!
# sollte sein weil 0 schon der erste punkt ist oder? also weil [x_min,x_max] statt normalerweise (x_min,x_max] oder?

# time
dt = 0.00001

t_min = 0
t_max = 0.5

n_t = int((t_max-t_min)/dt) +1 +1
# +1 to round up, second +1, because we also need to count 0

# globaly used steps for t
t_g = np.arange(0,n_t,dtype=dtype) * dt

# complete "mass" in the system (at start, if not continous)
initial_commulative_mass = 1

# +1 for the initial state


# diffusion parameter:
D_c = 1
# Task 2.9 
D_minus =  2
D_plus  =  1

# calculate and print out alpha
alpha = (D_c*dt)/dx**2/2
print(f"alpha = {alpha}")
print(f"Please note, my alpha differs by a factor of 1/2 from the alpha from the lecture.")
print("alpha lecture / 2 = my used alpha  ")
print(f"I adjusted this due to my derivation for Task 2.9 and to make alpha more comparable.")
