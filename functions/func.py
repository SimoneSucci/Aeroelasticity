import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from hipersim import MannTurbulenceField
from pathlib import Path
import sys


def get_pitch(time, switch1, switch2, pitch_value):
    if time<switch1 or time >switch2:
        theta_pitch = [7,7,7]
    else:
        theta_pitch = [pitch_value,pitch_value,pitch_value]
    return theta_pitch

def load_blade_data(txt_file: str
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Loads the blade data and records in an array for each characteristics"""

    blade_data = np.loadtxt(txt_file)
    radii = blade_data[:,0]
    chords = blade_data[:,2]
    betas = blade_data[:,1]
    thicknesses = blade_data[:,3]
    length = len(blade_data)

    return radii, chords, betas, thicknesses, length


def load_airfoils(thickness1_file: str, 
                  thickness2_file: str,
                  thickness3_file: str,
                  thickness4_file: str,
                  thickness5_file: str,
                  cylinder_file: str
                  )-> List:
    """Loads the airfoil data: CT and Cd for each airfoil shape. 
    All airfoils are then collected in a nested list."""

    airfoil1 = np.loadtxt(thickness1_file)
    airfoil2 = np.loadtxt(thickness2_file)
    airfoil3 = np.loadtxt(thickness3_file)
    airfoil4 = np.loadtxt(thickness4_file)
    airfoil5 = np.loadtxt(thickness5_file)
    airfoil6 = np.loadtxt(cylinder_file)
    airfoils = [airfoil1,airfoil2,airfoil3,airfoil4,airfoil5,airfoil6]

    return airfoils