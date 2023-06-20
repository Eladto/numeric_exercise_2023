import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import electron_mass,elementary_charge,epsilon_0,physical_constants
import scipy

ELECTRON_CHARGE = -elementary_charge
K = (4*np.pi*epsilon_0)**-1
HALF_RIB = 1
N = 200
TAU = 10**-3
STEPS = 1000


def return_to_square(point,half_rib):
    """
    Args:
        point: point array
        half_rib (float): half size of the rib of square 

    Returns:
        The point fixed in square
    """
    returned_point = np.min(np.array((point.reshape(2),np.array((HALF_RIB,HALF_RIB)))),0)
    returned_point = np.max(np.array((returned_point.reshape(2),np.array((-HALF_RIB,-HALF_RIB)))),0)
    return returned_point

def is_in_square(point,half_rib):
    """
    Args:
        point: point array
        half_rib (float): half size of the rib of square 

    Returns:
        True if the point square, either False
    """
    abs_point = np.abs(point)
    return abs_point[0]<=half_rib and abs_point[1]<=half_rib   

def get_uniformly_distributed_random_position_in_square(half_rib):
    """
    Args:
        half_rib (float): half size of the rib of square 

    Returns:
        Uniformly distributed random array positions in square
    """
    # get point in a square
    x = np.random.rand()*half_rib*2-half_rib
    y = np.random.rand()*half_rib*2-half_rib
    dist_from_origin = (x**2 + y**2)**0.5
    return (x,y)


def calc_coordinate_movement_by_field (start_position, start_velocity,field,time):
    """Calculate movement of electron

    Args:
        start_position : start position of electron
        start_velocity : start velocity of electron
        field : field vector in the position of electron
        time : time passed in this step

    Returns:
        array with the position of electron after movement
    """
    accelerate = (electron_mass**-1)*ELECTRON_CHARGE*field
    displacement = (start_velocity*time)+(0.5*accelerate)*time**2
    point_after_movement = start_position + displacement
    if(not is_in_square(point_after_movement.reshape(2),HALF_RIB)):
        point_after_movement = return_to_square(point_after_movement,HALF_RIB)
    return point_after_movement

def calculate_field_in_point(point,electron_positions):
    """
    Args:
        point : Vector of position
        electron_positions : Vectors of the positions of other electrons

    Returns:
        Field vector in position 
    """
    dist = scipy.spatial.distance.cdist(electron_positions,point)
    dist[dist == 0] = physical_constants["Bohr radius"][0]
    r_vectors = electron_positions+point*-1
    fields = np.multiply(-K*ELECTRON_CHARGE*r_vectors,np.power(dist,-3)) 
    return np.sum(fields,0)


def get_electron_position_after_movement(electron_position,other_electron_positions):
    """

    Args:
        electron_position: vector of electron position
        other_electron_positions : vector of the other electron positions

    Returns:
        electron position after movement, using the field calculated by the other electrons
    """
    poition_field = calculate_field_in_point(electron_position,other_electron_positions)
    return calc_coordinate_movement_by_field(electron_position,np.array((0,0)),poition_field,TAU)
    


# show section 2 graph - the distribution of electrons at the beginning
electron_positions = [get_uniformly_distributed_random_position_in_square(HALF_RIB) for i in range(N)]

x_positions = [position[0] for position in electron_positions]
y_positions = [position[1] for position in electron_positions]

ax = plt.axes()
ax.set_xlabel("x(m)")
ax.set_ylabel("y(m)")
ax.set_title("Random uniform distribution of electrons")
ax.scatter(x_positions,y_positions,s=8)
plt.show()








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

# show section 2 graph - the distribution of electrons after time
ax_electron_positions = plt.axes()
ax_electron_positions.set_xlabel("x(m)")
ax_electron_positions.set_ylabel("y(m)")
ax_electron_positions.set_title("Distribution of electrons after 1s")
ax_electron_positions.scatter(final_x_positions, final_y_positions,s=8)
plt.show()