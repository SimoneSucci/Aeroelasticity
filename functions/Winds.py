import numpy as np
from typing import List, Union, Tuple
from hipersim import MannTurbulenceField


def get_constant_wind(x: Union[float, np.ndarray],
                      V_hub: float, 
                      length: float
                  ) -> np.ndarray:
    """Outputs the velocity vector for a constant wind velocity in the z direction."""
    V0_array = np.ones(len(x))*V_hub
    return np.array([np.zeros(length),np.zeros(length),V0_array])

def get_wind_shear(x: Union[float, np.ndarray], 
               V_hub: float, 
               H: float, 
               nu: float
               ) -> np.ndarray:
    """Outputs the velocity vector in the case of wind shear. 
    x should be the vertical position in frame 1."""

    v_shear = np.array([np.zeros_like(x),np.zeros_like(x),V_hub*(x/H)**nu])
    return v_shear
    
def get_tower_speed(V0: Union[float, np.ndarray], 
                coord: np.ndarray, 
                a_tower: float, 
                H: float
                ) -> np.ndarray:
    """Outputs velocity array at each point in coord: for input wind V0, and with tower shadow effect."""

    V0  = V0[2]
    x, y, z = coord[0], coord[1], coord[2]
    if np.all(x)<H:
        a = a_tower
    else:
        a=0
    r = np.sqrt(y**2+z**2)
    Vr = z/r*V0*(1-(a/r)**2)
    Vt = y/r*V0*(1+(a/r)**2)
    
    Vel = np.array([np.zeros_like(x),y/r*Vr-z/r*Vt, z/r*Vr+y/r*Vt ])

    return Vel

def build_turbulence_box(Nxyz_input, dxyz_input, U_mean) -> None:

# Generate a Mann box, scale it to a certain TI and mean wind speed, and save it to a file.
    mann_box = MannTurbulenceField.generate(Nxyz=Nxyz_input, dxyz = dxyz_input, L=33.6, Gamma=3.9)
    mann_box.scale_TI(TI=0.1, U=U_mean)
    mann_box.to_netcdf(filename = "mann_box_try1.nc")
         # Load the file.
    mann_box = MannTurbulenceField.from_netcdf("mann_box_try1.nc")

    # Transform the Mann box to a `DataArray` (from the package `xarray`)
    ds_mann_box = mann_box.to_xarray()
    return ds_mann_box

def interpolate_turbulence_box(ds_mann_box, position: np.ndarray, length: float, H: float, V_hub, t):


    # Example of how to interpolate to a single point.
    # Interpolating to lists of x, y, z, results in interpolation to a grid of those values.
    # For interpolation to specific points at once, look into the documentation (or ask your friendly LLM).
    xcoord = position[0,:] + H -  np.ones(length)*ds_mann_box.y.max().values/2
    ycoord = position[1,:] + np.ones(length)*ds_mann_box.x.max().values/2
    zcoord = -position[2,:] + V_hub*t
    #uvw_interp = ds_mann_box.interp(x=xcoord, y=ycoord, z=zcoord, method = 'linear').data  # shape (3,) ie (u, v, w)
    points_ds = ds_mann_box.interp(
    x=("point", xcoord),
    y=("point", ycoord),
    z=("point", zcoord),
    method="linear"
    )
    uvw_interp = points_ds.data 

    # Example of how to select the `u` component of the turbulent fluctuations and
    # and calculating the TI of `u`.
   # u_turb = uvw_interp.sel(uvw="u")
    #TI = u_turb.std("x") / U_mean
    return uvw_interp