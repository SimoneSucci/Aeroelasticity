# Assignment 1

For now, this only shows a pure Python implementation of how turbulent fluctuations can be created. The rest of the code will be shown after the submission of the first assignment.

## Installation

The only requirements are the packages `hipersim` and `wetb`. You can install them individually or by `pip install -r requirements.txt`.

## Side notes

- With this code, you calculate the turbulent *fluctuations*. What should their means be? If they are not what you expect, the resolution and size of your Mann box is most likely too small.
- The `to_xarray()` function returns a `DataArray` of the package `xarray`. Very simply, `xarray` is like `numpy` but with "labelled" axes. I.e., you don't need to remember which dimension of a numpy array corresponds to what, but you can do things like `da.sel(x=119, y=-100, z=10)` to select (`sel`) whatever data you have at x=119, y=-100, z=10. Here, `da` is a `xarray.DataArray`. Additionally, these data arrays already implement functions to interpolate, get the standard deviation, the mean, and much more.
