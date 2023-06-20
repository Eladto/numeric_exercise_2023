import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.constants import electron_mass,elementary_charge 

ELECTRON_CHARGE = -elementary_charge
INIT_TIME = 0
INIT_X = 0
INIT_Y = 0
NUM_OF_TAUS = 100
E0 = 30
TAU = 10**-15
RANDOM_V0_SIZE = 0.002

def calc_y (time,velocity_y,y0):
    """calcualte y position by starting conditions

    Args:
        time (float): time passed 
        velocity_y (float): starting y direction velocity
        y0 (float): starting y position

    Returns:
        float: y position after time has passed
    """
    return  y0+(velocity_y*time)


def calc_x (time,velocity_x,x0):
    """calcualte x position by starting conditions

    Args:
        time (float): time passed 
        velocity_x (float): starting x direction velocity
        x0 (float): starting x position


    Returns:
        float: x position after time has passed
    """
    accelerate = -(electron_mass**-1)*ELECTRON_CHARGE*E0
    return  x0+(velocity_x*time)+(0.5*accelerate)*time**2

def pol2cart(rho, phi):
    """transform polarized coordinates to cartesian coordinates

    Args:
        rho (float): radius
        phi (float): angle
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_random_velocity():
    """
    Returns:
        vector with cartesian coordinated random velocity, which its size is RANDOM_V0_SIZE
    """
    rho = RANDOM_V0_SIZE
    phi = np.random.rand()*2*np.pi
    return np.array(pol2cart(rho,phi))

def get_electron_route(i):
    """calculate electron route

    Args:
        i (int): electron number

    Returns:
        tuple(list(float)): Pair of lists, represent the positions of the electron during time
    """
    x_points_list = [INIT_X]
    y_points_list = [INIT_Y]
    pre_x = INIT_X
    pre_y = INIT_Y

    for interval in range(NUM_OF_TAUS+1):
        velocity = get_random_velocity()
        current_x = calc_x(TAU,velocity[0],pre_x)
        current_y = calc_y(TAU,velocity[1],pre_y)
        x_points_list.append(current_x)
        y_points_list.append(current_y)
        pre_x = current_x
        pre_y = current_y

    print("final velocity {}: {}".format(i+1,x_points_list[-1]/(NUM_OF_TAUS*TAU)))
    return (x_points_list,y_points_list)


def scaller_foramtter():
    # Create a ScalarFormatter with the desired format
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_scientific(True)
    return formatter

def customize_plot(plt):
     # Customize the plot
    plt.title("Elecrtrons Routes")
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.grid()
    # Get the current axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(scaller_foramtter())
    ax.xaxis.set_major_formatter(scaller_foramtter())
    plt.tight_layout()

def create_routes_and_plot():
    # Create the figure and subplots
    for i in range(3):
        (x,y) = get_electron_route(i)
        # Plot scatter graph
        plt.plot(x, y)
    
    customize_plot(plt)
    plt.show()

create_routes_and_plot()
