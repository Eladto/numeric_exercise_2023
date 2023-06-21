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
STEPS = 1000



def is_in_disk(point,radius):
    """
    Args:
        point: point array
        radius (float): radius of ball

    Returns:
        True if the point in the radius of disk, either False
    """
    return (np.sum(point**2))**0.5 <= radius


def is_on_circle(point,radius):
    """
    Args:
        point: point array
        radius (float): radius of sphere

    Returns:
        True if the point on the circle, either False
    """
    return (np.sum(point**2))**0.5 == radius


def get_uniformly_distributed_random_position_in_disk(radius):
    """
    Args:
        radius (float): radius of ball

    Returns:
        Uniformly distributed random array position in disk
    """
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

def return_to_circle(point,radius):
    """Normalize the point, so it will be on the circle

    Args:
        point: point array
        radius (float): radius of sphere

    Returns:
        Point on the angle(in polarized coordinates), but on the circle 
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
    if (not is_in_disk(point_after_movement,RADIUS)):
        point_after_movement = return_to_circle(point_after_movement,RADIUS)
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
    position_field = calculate_field_in_point(electron_position,other_electron_positions)
    return calc_coordinate_movement_by_field(electron_position,np.array((0,0)),position_field,TAU)

def get_elget_electrons_densityectrons_density(electron_positions, bins):
    """_summary_

    Args:
        electron_positions : The positions of electrons
        bins : The disks radiuses to count electrons on

    Returns:
        tuple: The count of electrons on every radius, radiuses
    """
    dist = np.sum(np.array(electron_positions)**2,1)**0.5
    return np.histogram(dist,bins)


def scaller_foramtter():
    # Create a ScalarFormatter with the desired format
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_scientific(True)
    return formatter

def analytic_solution(r1,jump):
    """

    Args:
        r1 (float): the lower radius of disk
        jump (float): the difference between the radiuses of disk

    Returns:
        the density of charge by the analytic solution
    """
    r2 = r1+jump
    return ((-3.2*10**-17)/np.pi)*((1-r1**2)**0.5-(1-r2**2)**0.5)/(r2**2-r1**2)

electron_positions = [get_uniformly_distributed_random_position_in_disk(RADIUS) for i in range(N)]

x_positions = [position[0] for position in electron_positions]
y_positions = [position[1] for position in electron_positions]

# show section 1 graph - the distribution of electrons at the beginning


ax = plt.axes()
ax.scatter(x_positions,y_positions,s=12)
ax.set_xlabel("x(m)")
ax.set_ylabel("y(m)")
ax.set_title("Random uniform distribution of electrons")
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

# show section 1 graph - the distribution of electrons after time
ax_electron_positions = plt.axes()
ax_electron_positions.scatter(final_x_positions, final_y_positions,s=12)
ax_electron_positions.set_xlabel("x(m)")
ax_electron_positions.set_ylabel("y(m)")
ax_electron_positions.set_title("Distribution of electrons after 1s")
plt.show()

# show section 3 graph - the density of electrons
electron_density,radius = get_electrons_density(electron_positions,np.arange(11)*0.1)
area_by_radius = np.pi*radius**2
ring_area = area_by_radius[1:] -  area_by_radius[:-1]
ax_density = plt.axes()
ax_density.set_title("Density as function of r")
ax_density.set_xlabel("r(m)")
ax_density.set_ylabel("density(c/m^2)")

ax_density.scatter(np.arange(10)*0.1,electron_density*ELECTRON_CHARGE/ring_area,s=8,c="green")
ax_density.plot(np.arange(10)*0.1,list(map(lambda x: analytic_solution(x,0.1),list(np.arange(10)*0.1))),'.--')
ax_density.yaxis.set_major_formatter(scaller_foramtter())
plt.show()
