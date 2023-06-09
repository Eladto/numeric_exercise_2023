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



def is_in_disk(point,radius):
    return (np.sum(point**2))**0.5 < radius


def is_in_circle(point,radius):
    return (np.sum(point**2))**0.5 == radius


def get_uniformly_distributed_random_position_in_disk(radius):
    # get point in a square
    x = np.random.rand()*radius*2-radius
    y = np.random.rand()*radius*2-radius
    dist_from_origin = (x**2 + y**2)**0.5
    # reject if points not in the ball
    while dist_from_origin>radius:
        # get point in a square
        x = np.random.rand()*radius
        y = np.random.rand()*radius
        dist_from_origin = (x**2 + y**2)**0.5
    return (x,y)

def get_displacement_to_circle(point,displacement,radius):
    point = point.reshape(2)
    a = (displacement[0]**2+displacement[1]**2)
    b = 2*np.sum(np.multiply(point,displacement))
    c = point[0]**2+point[1]**2-radius**2
    factor = (-b+(b**2-4*a*c)**0.5)/(2*a)
    return factor*displacement



def calc_coordinate_movement_by_field (start_position, start_velocity,field,time):
    accelerate = (electron_mass**-1)*ELECTRON_CHARGE*field
    displacement = (start_velocity*time)+(0.5*accelerate)*time**2
    point_after_movement = start_position+displacement
    if (not is_in_disk(point_after_movement,RADIUS)):
        displacement_to_sphere = get_displacement_to_circle(start_position,displacement,RADIUS)
        point_after_movement = start_position + displacement_to_sphere
    
    

    return point_after_movement

electron_positions = [get_uniformly_distributed_random_position_in_disk(RADIUS) for i in range(N)]

x_positions = [position[0] for position in electron_positions]
y_positions = [position[1] for position in electron_positions]

ax = plt.axes()
ax.scatter(x_positions,y_positions)
plt.show()


def calculate_field_in_point(point,electron_positions):
    dist = scipy.spatial.distance.cdist(electron_positions,point)
    r_vectors = electron_positions+point*-1
    fields = np.multiply(-K*ELECTRON_CHARGE*r_vectors,np.power(dist,-3)) 
    return np.sum(fields,0)


def get_electron_position_after_movement(electron_position,other_electron_positions):
    poition_field = calculate_field_in_point(electron_position,other_electron_positions)
    return calc_coordinate_movement_by_field(electron_position,np.array((0,0)),poition_field,TAU)
    


for step in range(STEPS):
    new_electron_positions = []
    for i in range(len(electron_positions)):
        electron_position = np.array(electron_positions[i]).reshape(1,2)
        other_electron_positions = np.array(electron_positions[:i]+electron_positions[i+1:])
        new_electron_position = get_electron_position_after_movement(electron_position,other_electron_positions)
        new_electron_positions.append(new_electron_position.reshape(2))
    electron_positions = new_electron_positions


final_x_positions = [position[0] for position in electron_positions]
final_y_positions = [position[1] for position in electron_positions]

# show section 1 graph - the distribution of electrons after time
ax_electron_positions = plt.axes()
ax_electron_positions.scatter(final_x_positions, final_y_positions)
plt.show()