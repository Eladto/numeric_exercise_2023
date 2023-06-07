import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import electron_mass,elementary_charge,epsilon_0
import scipy
from mpl_toolkits import mplot3d


ELECTRON_CHARGE = -elementary_charge
K = (4*np.pi*epsilon_0)**-1
RADIUS = 1
N = 200
TAU = 10**-3
STEPS = 1000


def get_uniformly_distributed_random_position_in_sphere(radius):
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


def calc_coordinate_movement_by_field (start_position, start_velocity,field,time):
    accelerate = (electron_mass**-1)*ELECTRON_CHARGE*field
    return  start_position+(start_velocity*time)+(0.5*accelerate)*time**2


def get_uniformly_distributed_random_position_in_ball(radius):
    # get point in a cube
    x = np.random.rand()*radius*2-radius
    y = np.random.rand()*radius*2-radius
    z = np.random.rand()*radius*2-radius
    normal = (x**2 + y**2 + z**2)**0.5
    # reject if points not in the ball
    while normal>radius:
         # get point in a cube
        x = np.random.rand()*radius
        y = np.random.rand()*radius
        z = np.random.rand()*radius
        normal = (x**2 + y**2 + z**2)**0.5
    return (x,y,z)

def calculate_field_in_point(point,electron_positions):
    dist = scipy.spatial.distance.cdist(electron_positions,point)
    r_vectors = electron_positions+point*-1
    fields = np.multiply(K*ELECTRON_CHARGE*r_vectors,np.power(dist,-3))
    return np.sum(fields)
    

def get_electron_position_after_movement(electron_position,other_electron_positions):
    poition_field = calculate_field_in_point(electron_position,other_electron_positions)
    return calc_coordinate_movement_by_field(electron_position,np.array((0,0,0)),poition_field,TAU)
    

electron_positions = [get_uniformly_distributed_random_position_in_ball(RADIUS) for i in range(N)]

x_positions = [position[0] for position in electron_positions]
y_positions = [position[1] for position in electron_positions]
z_positions = [position[2] for position in electron_positions]

ax = plt.axes(projection='3d')
ax.scatter(x_positions,y_positions,z_positions, c=z_positions,cmap='viridis', linewidth=0.5)
plt.show()

for step in range(STEPS):
    new_electron_positions = []
    for i in range(len(electron_positions)):
        electron_position = np.array(electron_positions[i]).reshape(1,3)
        other_electron_positions = np.array(electron_positions[:i]+electron_positions[i+1:])
        new_electron_position = get_electron_position_after_movement(electron_position,other_electron_positions)
        new_electron_positions.append(electron_position)
    electron_position = new_electron_positions

final_x_positions = [position[0] for position in electron_positions]
final_y_positions = [position[1] for position in electron_positions]
final_z_positions = [position[2] for position in electron_positions]

ax = plt.axes(projection='3d')
ax.scatter(final_x_positions, final_y_positions, final_z_positions, c=final_z_positions,cmap='viridis', linewidth=0.5)
plt.show()
