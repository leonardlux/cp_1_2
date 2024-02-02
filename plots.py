import config as c
import numpy as np
from ipywidgets import interact
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib import cm


# Plots (simpel)

t_diff_times = [0, 1, 5, 10, 25,50, 100]
def plot_diff_times(
        U,
        title="",
        t=c.t_g,
        t_diff_times = t_diff_times
        ):
    fig = plt.figure()
    plt.title(title)
    for t in t_diff_times:
        plt.plot(c.x_g,U[t], label=f"t={c.dt*t}")
    #plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x)")
    #plt.yscale("log")
    plt.show()

def plot_mass(
        U,
        title="",
        t=c.t_g
        ):
    # sum up mass
    Mass = np.sum(U,axis=1)
    Mass_rel = Mass/Mass[0]*100
    fig = plt.figure(1, figsize = (10,6))
    plt.title(title)
    plt.plot(c.t_g,Mass_rel,label="mass")
    plt.xlabel("time")
    plt.ylabel("$m/m_0$ in % ")
    #plt.yscale("log")
    plt.axhline(100,label="starting value",alpha=0.2,c="grey")
    plt.ylim(-0.1,105,)
    plt.xlim(c.t_min,c.t_max)
    plt.legend()
    plt.show()

def plot_masses(
        U_sims,
        t=c.t_g,
        procent=True,
        t_i_max=c.n_t-1,
        log_scale=False
        ):
    # sum up mass
    t = c.t_g[:t_i_max]
    fig = plt.figure(1, figsize = (10,6))
    plt.title("All mass plots")
    for U_i in U_sims:
        Mass = np.sum(U_i.U[:t_i_max],axis=1)
        if procent:
            Mass = Mass/Mass[0]*100
        plt.plot(t,Mass,label=U_i.title)
    plt.xlabel("time")
    if procent:
        plt.ylabel("$m/m_0$ in % ")
    else:
        plt.ylabel("$m$")
    if log_scale:
        plt.yscale("log")
    plt.xlim(t[0],t[-1])
    plt.legend()
    plt.show()


# Plots (complex)
def sketch(
        U,
        t_i_steps=100,
        t_i_max = c.n_t-1,
        log_scale=False
        ):
    def plotter(t):
        t = int(t)
        if log_scale:  
            plt.yscale("log")
        fig = plt.figure(1, figsize = (10,6))
        plt.plot(c.x_g,U[t],label=f"at $t={t*c.dt}$")
        plt.legend()
    slider = interact(plotter, t = (0, t_i_max, t_i_steps))

def sketches(
        U_sims,
        t_i_steps=100,
        t_i_max = c.n_t-1,
        log_scale=False
        ):
    def plotter(t):
        t = int(t)
        if log_scale:  
            plt.yscale("log")
        fig = plt.figure(1, figsize = (10,6));
        for U_sim in U_sims:
            plt.plot(c.x_g,U_sim.U[t],label=U_sim.title,alpha=0.9);

        plt.title(f"at $t={t*c.dt}$")
        plt.legend()
    interact(plotter, t = (0, t_i_max, t_i_steps));
    
def animate(U):
    print(U)
    fig, ax = plt.subplots()
    plt.ylim(0,np.max(U[0]))
    line, = ax.plot(c.x_g,U[int(0)])

    def animate(t):
        ax.set_ylim(np.max(U[int(t)]*1.3))
        line.set_ydata(U[int(t)])
        return line

    ani = animation.FuncAnimation(fig, animate, repeat=True, frames=int((c.n_t-1)/400))

    from IPython.display import HTML
    return HTML(ani.to_jshtml())

def plot_3d(U):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    t_step_size = int((len(c.t_g)-1)/(len(c.x_g)-1))

    offset = 1

    X, Y = np.meshgrid(
        c.x_g[:-1][offset:], 
        c.t_g[t_step_size::t_step_size][offset:]
        )
    Z = U.U
    Z = Z[:-1][::t_step_size]
    Z = Z[offset:,offset+1:]
    Z = np.swapaxes(Z,0,1)


    # Plot the surface.
    surf = ax.plot_surface( 
        X, 
        Y, 
        Z,
        cmap=cm.coolwarm,
        antialiased=False, 
        lw=0.5, 
        rstride=8, 
        cstride=8,
        alpha=0.6)

    # # Customize the z axis.
    plt.title(U.title)
    ax.set_zlim(0, 3)
    ax.contourf(X, Y, Z, zdir='x', offset=1, cmap='coolwarm')

    # Add a color bar which maps values to colors.
    

    plt.show()