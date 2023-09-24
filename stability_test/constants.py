from openmm import unit
from openmm.unit import Quantity
from openmmtools.constants import kB

# define units
distance_unit = unit.angstrom
time_unit = unit.femto * unit.seconds
speed_unit = distance_unit / time_unit

# define simulation parameters
stepsize = Quantity(1, time_unit)

# define simulation parameters
collision_rate = 1 / Quantity(1, unit.pico * unit.second)
temperature = Quantity(300, unit.kelvin)
pressure = Quantity(1, unit.atmosphere)

kBT = kB * temperature

available_nnps_and_implementation = [
    ("ani2x", "nnpops"),
    ("ani2x", "torchani"),
    ("ani1ccx", "nnpops"),
    ("ani1ccx", "torchani"),
]
