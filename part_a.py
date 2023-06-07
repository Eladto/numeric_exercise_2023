import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import electron_mass,elementary_charge 

INIT_TIME = 0
INIT_X = 0
INIT_Y = 0
NUM_OF_TAUS = 100
E0 = 30
TAU = 10**-15
RANDOM_V0_SIZE = 0.002

def calc_y_where_only_field (t,v_y,y0):
    return  y0+(v_y*t)


def calc_x_where_only_field (t,v_x,x0):
    accelerate = (electron_mass**-1)*elementary_charge*E0
    return  x0+(v_x*t)+(0.5*accelerate)*t**2

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_random_velocity():
    rho = RANDOM_V0_SIZE
    phi = np.random.rand()*2*np.pi
    return np.array(pol2cart(rho,phi))

def get_electron_route(i):
    x_points_list = [INIT_X]
    y_points_list = [INIT_Y]
    pre_x = INIT_X
    pre_y = INIT_Y

    for interval in range(NUM_OF_TAUS+1):
        velocity = get_random_velocity()
        current_x =calc_x_where_only_field(TAU,velocity[0],pre_x)
        current_y = calc_y_where_only_field(TAU,velocity[1],pre_y)
        x_points_list.append(current_x)
        y_points_list.append(current_y)
        pre_x = current_x
        pre_y = current_y

    print("final velocity {}: {}".format(i+1,x_points_list[-1]/(NUM_OF_TAUS*TAU)))
    return (x_points_list,y_points_list)


def create_routes_and_plot():
    # Create the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i in range(axes.shape[0]):
        ax = axes[i]
        (x,y) = get_electron_route(i)
        # Plot scatter graph
        ax.plot(x, y)
        # Customize the plot (optional)
        ax.set_title("elecrtron {} route".format(i+1))
        ax.grid()

    plt.tight_layout()
    plt.show()

create_routes_and_plot()
