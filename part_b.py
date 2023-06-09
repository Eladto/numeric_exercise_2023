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


def vector_size(x,y,z):
    return (x**2 + y**2 + z**2)**0.5

def get_uniformly_distributed_random_position_in_sphere(radius):
    x = np.random.rand()
    y = np.random.rand()
    z = np.random.rand()
    while (x == 0 and y ==0 and z == 0):
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
    
    size = vector_size(x,y,z)
    x = (x/ size)*radius
    y = (y/ size)*radius
    z = (z/ size)*radius
    return x,y, z

def is_in_ball(point,radius):
    return (np.sum(point**2))**0.5 < radius

def is_in_sphere(point,radius):
    return (np.sum(point**2))**0.5 == radius

def get_displacement_to_sphere(point,displacement,radius):
    point = point.reshape(3)
    a = (displacement[0]**2+displacement[1]**2+displacement[2]**2)
    b = 2*np.sum(np.multiply(point,displacement))
    c = point[0]**2+point[1]**2+point[2]**2-radius**2
    factor = (-b+(b**2-4*a*c)**0.5)/(2*a)
    return factor*displacement

def calc_coordinate_movement_by_field (start_position, start_velocity,field,time):
    accelerate = (electron_mass**-1)*ELECTRON_CHARGE*field
    displacement = (start_velocity*time)+(0.5*accelerate)*time**2
    point_after_movement = start_position+displacement
    if (not is_in_ball(point_after_movement,RADIUS)):
        displacement_to_sphere = get_displacement_to_sphere(start_position,displacement,RADIUS)
        point_after_movement = start_position + displacement_to_sphere
    
    

    return point_after_movement


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
    fields = np.multiply(-K*ELECTRON_CHARGE*r_vectors,np.power(dist,-3))
    return np.sum(fields,0)

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
not_in_sphere_percentages = []
for step in range(STEPS):
    not_in_sphere_counter = 0
    new_electron_positions = []
    for i in range(len(electron_positions)):
        electron_position = np.array(electron_positions[i]).reshape(1,3)
        not_in_sphere_counter += int(not is_in_sphere(electron_position,RADIUS))
        other_electron_positions = np.array(electron_positions[:i]+electron_positions[i+1:])
        new_electron_position = get_electron_position_after_movement(electron_position,other_electron_positions)
        new_electron_positions.append(new_electron_position.reshape(3))
    not_in_sphere_percentages.append(not_in_sphere_counter/N)
    electron_positions = new_electron_positions

final_x_positions = [position[0] for position in electron_positions]
final_y_positions = [position[1] for position in electron_positions]
final_z_positions = [position[2] for position in electron_positions]

ax_electron_positions = plt.axes(projection='3d')
ax_electron_positions.scatter(final_x_positions, final_y_positions, final_z_positions, c=final_z_positions,cmap='viridis', linewidth=0.5)
plt.show()

ax_electron_percentages =  plt.axes()
times = np.array(range(STEPS))*TAU
ax_electron_percentages.scatter(times,not_in_sphere_percentages)
plt.show()
