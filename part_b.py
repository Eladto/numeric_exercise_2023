import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.constants import electron_mass,elementary_charge,epsilon_0
import scipy
from mpl_toolkits import mplot3d


ELECTRON_CHARGE = -elementary_charge
K = (4*np.pi*epsilon_0)**-1
RADIUS = 1
N = 200
TAU = 10**-3
STEPS = 20


def is_in_ball(point,radius):
    """
    Args:
        point: point array
        radius (float): radius of ball

    Returns:
        True if the point in the radius of ball, either False
    """
    return (np.sum(point**2))**0.5 <= radius

def is_in_sphere(point,radius):
    """
    Args:
        point: point array
        radius (float): radius of sphere

    Returns:
        True if the point on the sphere, either False
    """
    return np.round((np.sum(point**2))**0.5,10) == radius


def return_to_sphere(point,radius):
    """Normalize the point, so it will be on the sphere

    Args:
        point: point array
        radius (float): radius of sphere

    Returns:
        Point on the angle(in polarized coordinates), but on the sphere 
    """
    dist = (np.sum(point**2))**0.5
    return point*(radius/dist)

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
    point_after_movement = start_position+displacement
    if (not is_in_ball(point_after_movement,RADIUS)):
        point_after_movement = return_to_sphere(point_after_movement,RADIUS)
    return point_after_movement


def get_uniformly_distributed_random_position_in_ball(radius):
    """
    Args:
        radius (float): radius of ball

    Returns:
        Uniformly distributed random array position in ball
    """
    # get point in a cube
    x = np.random.rand()*radius*2-radius
    y = np.random.rand()*radius*2-radius
    z = np.random.rand()*radius*2-radius
    normal = (x**2 + y**2 + z**2)**0.5
    # Reject if points not in the ball
    while normal>radius:
         # get another point in a cube
        x = np.random.rand()*radius
        y = np.random.rand()*radius
        z = np.random.rand()*radius
        normal = (x**2 + y**2 + z**2)**0.5
    return (x,y,z)

def calculate_field_in_point(point,electron_positions):
    """

    Args:
        point : Vector of position
        electron_positions : Vectors of the positions of other electrons

    Returns:
        Field vector in position 
    """
    # vector of distances between position and electrons 
    dist = scipy.spatial.distance.cdist(electron_positions,point)
    # vectors of difference between electron positions
    r_vectors = electron_positions+point*-1
    # Using the formulae -(kqr)/(|r|^3), calculate the field in point
    fields = np.multiply(-K*ELECTRON_CHARGE*r_vectors,np.power(dist,-3)) 
    return np.sum(fields,0)

def calculate_potential_in_point(point,electron_positions):
    """

    Args:
        point : Vector of position
        electron_positions : Vectors of the positions of other electrons

    Returns:
        Potential in point
    """
    # vector of distances between position and electrons 
    dist = scipy.spatial.distance.cdist(electron_positions,point)
    # Using the formula -kq/|r|
    potentials = np.multiply(-K*ELECTRON_CHARGE,np.power(dist,-1)) 
    return np.sum(potentials)


def get_electron_position_after_movement(electron_position,other_electron_positions):
    """

    Args:
        electron_position: vector of electron position
        other_electron_positions : vector of the other electron positions

    Returns:
        electron position after movement, using the field calculated by the other electrons
    """
    poition_field = calculate_field_in_point(electron_position,other_electron_positions)
    return calc_coordinate_movement_by_field(electron_position,np.array((0,0,0)),poition_field,TAU)
    
def scaller_foramtter():
    # Create a ScalarFormatter with the desired format
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_scientific(True)
    return formatter


electron_positions = [get_uniformly_distributed_random_position_in_ball(RADIUS) for i in range(N)]

x_positions = [position[0] for position in electron_positions]
y_positions = [position[1] for position in electron_positions]
z_positions = [position[2] for position in electron_positions]

# Show the electrons distirbution at the beginning
ax = plt.axes(projection='3d')
ax.scatter(x_positions,y_positions,z_positions, c=z_positions,cmap='viridis', linewidth=0.5)
ax.set_xlabel("x(m)")
ax.set_ylabel("y(m)")
ax.set_zlabel("z(m)")
plt.show()
# the percentages of electrons not in the sphere for every step
not_in_sphere_percentages = []
for step in range(STEPS):
    not_in_sphere_counter = 0
    new_electron_positions = []
    for i in range(len(electron_positions)):
        electron_position = np.array(electron_positions[i]).reshape(1,3)
        other_electron_positions = np.array(electron_positions[:i]+electron_positions[i+1:])
        new_electron_position = get_electron_position_after_movement(electron_position,other_electron_positions)
        not_in_sphere_counter += int(not is_in_sphere(electron_position,RADIUS))
        new_electron_positions.append(new_electron_position.reshape(3))
    # print(not_in_sphere_counter)
    not_in_sphere_percentages.append(not_in_sphere_counter/N)
    electron_positions = new_electron_positions

final_x_positions = [position[0] for position in electron_positions]
final_y_positions = [position[1] for position in electron_positions]
final_z_positions = [position[2] for position in electron_positions]

# show section 2 graph - the distribution of electrons after time
ax_electron_positions = plt.axes(projection='3d')
ax_electron_positions.set_xlabel("x(m)")
ax_electron_positions.set_ylabel("y(m)")
ax_electron_positions.set_zlabel("z(m)")
ax_electron_positions.scatter(final_x_positions, final_y_positions, final_z_positions, c=final_z_positions,cmap='viridis', linewidth=0.5)
plt.show()

# show section 3 graph - the percentage not in sphere as function of time
ax_electron_percentages =  plt.axes()
times = np.array(range(STEPS))*TAU
ax_electron_percentages.set_label("Percentages of electrons in the ball as function of time")
ax_electron_percentages.set_xlabel("time(s)")
ax_electron_percentages.set_ylabel("percentages")
ax_electron_percentages.scatter(times,np.array(not_in_sphere_percentages)*100)
plt.show()

# show section 4 graph - the potential as function of r
r_samples = np.array(range(0,100))*0.1
potentials =  np.array([calculate_potential_in_point(np.array((r,0,0)).reshape(1,3),electron_positions) for r in r_samples])
ax_r_potentials = plt.axes()
ax_r_potentials.set_label("Potential as function of r")
ax_r_potentials.set_xlabel("time(s)")
ax_r_potentials.set_ylabel("potential(v)")
ax_r_potentials.yaxis.set_major_formatter(scaller_foramtter())
ax_r_potentials.xaxis.set_major_formatter(scaller_foramtter())
ax_r_potentials.scatter(r_samples,potentials)
plt.show()
