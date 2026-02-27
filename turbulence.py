from hipersim import MannTurbulenceField


# Generate a Mann box, scale it to a certain TI and mean wind speed, and save it to a file.
mann_box = MannTurbulenceField.generate(Nxyz=(16, 16 , 500), dxyz=(5, 5, 5), L=33.6, Gamma=3.9)
U_mean = 10
mann_box.scale_TI(TI=0.1, U=U_mean)
mann_box.to_netcdf(filename="mann_box.nc")

# Load the file.
mann_box = MannTurbulenceField.from_netcdf("mann_box.nc")
uvw = mann_box.uvw  # shape (3, 512, 32, 16), ie u = uvw[0]

# Transform the Mann box to a `DataArray` (from the package `xarray`)
ds_mann_box = mann_box.to_xarray()

# Example of how to interpolate to a single point.
# Interpolating to lists of x, y, z, results in interpolation to a grid of those values.
# For interpolation to specific points at once, look into the documentation (or ask your friendly LLM).
uvw_interp = ds_mann_box.interp(x=500, y=85, z=11).data  # shape (3,) ie (u, v, w)

# Example of how to select the `u` component of the turbulent fluctuations and
# and calculating the TI of `u`.
u_turb = ds_mann_box.sel(uvw="u")
TI = u_turb.std("x") / U_mean
# This prints the TI that the specific y, z points sees over time. The return is still a DataArray.
print(TI.sel(y=20, z=40))
# If you just want the TI data there, do this
print(TI.sel(y=20, z=40).data)
