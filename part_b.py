import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import electron_mass,elementary_charge 

RADIUS = 2
N = 200
TAU = 10**-3


def get_random_position_in_sphere(radius):
    x = np.random.rand()
    y = np.random.rand()
    z = np.random.rand()
    while (x == 0 and y ==0 and z == 0):
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
    
    normal = (x**2 + y**2 + z**2)**0.5
    x = (x/ normal)*radius
    y = (y/ normal)*radius
    z = (z/ normal)*radius
    return x,y, z

electron_positions = [get_random_position_in_sphere(RADIUS) for i in range(N)]
x_positions = [position[0] for position in electron_positions]
y_positions = [position[1] for position in electron_positions]
z_positions = [position[2] for position in electron_positions]
