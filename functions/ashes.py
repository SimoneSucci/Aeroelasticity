import re
import numpy as np
import pandas as pd
import xarray as xr
import pathlib
from pathlib import Path



def import_results_timesteps (file_path):
    """Read the simulation results from a .txt file exported from Ashes.

    Args:
        file_path ((Union[str | Path])): File path of the simulation results
            .txt file

    Returns:
        data_df (pandas dataframe): Dataframe with the simulation timeseries
        unit_dict (dict): Dictionary with the units for each sensor
    """
    # Check file path
    file_path = check_fpath(file_path)

    # Find the row with the variable names and the first row with the results
    header_row = None
    content_start_row = None

    headers = None
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if i>100:
                raise ValueError("No valid header information found in the "
                                 + "first 100 lines. Aborting file import")
            line = line.strip()

            # Find row with variable names (two lines below the line which
            # starts with "# Column")
            if line.startswith("# Column"):
                header_row = i+2

            # Extract headers and units
            if i == header_row:
                raw_headers = line.strip().split('\t')
                unit_dict = {re.sub(r"\s+\[.*\]", "", s).strip()
                             : re.search(r"(?:\s+\[)(.*)(?:\])", s).group(1)
                             for s in raw_headers}
                headers = [re.sub(r"\s+\[.*\]", "", s).strip()
                           for s in raw_headers]

            # Once the headers have been find, look for the first row with
            # numerical values separated by tabspaces or whitespaces
            if headers is not None and content_start_row is None:
                if re.match(r"^(\d+\.?\d*(\s+|\t))+", line):
                    content_start_row = i
                    break

    data = np.loadtxt(file_path, skiprows=content_start_row)
    data_df = pd.DataFrame(columns = headers, data=data)


    return data_df, unit_dict

def import_results_bld_spanwise (file_path):
    """Read the blade spanwise sensor simulation results from the .txt
    file exported from Ashes.

    Args:
        file_path (Union[str | Path]): File path of the simulation results
            .txt file

    Returns:
        data_ds (xarray dataset): Dataset with the blade sensors as variables
            and the timestamps and rotor positions as coordinates
        times (numpy array): Time series of the simulation
        unit_dict (dictionary): Dictionary with the sensor names as keys and
            the respective units as the values
    """
    # Check file path
    file_path = check_fpath(file_path)

    # Find the row with the variable names, the spanwise sensor positions and
    # the first row with the results
    header_row = None
    stations_row = None
    content_start_row = None

    headers = None
    r = None
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if i>100:
                raise ValueError("No valid header information found in the "
                                 "first 100 lines. Aborting file import")
            line = line.strip()

            # Find row with variable names (two lines below the line which
            # starts with "# Column")
            if line.startswith("# Column"):
                header_row = i+2

            # Extract headers and units
            if i == header_row:
                raw_headers = line.strip().split('\t')
                unit_dict = {re.sub(r"\s+\[.*\]", "", s).strip()
                             : re.search(r"(?:\s+\[)(.*)(?:\])", s).group(1)
                             for s in raw_headers}
                headers = [re.sub(r"\s+\[.*\]", "", s).strip()
                           for s in raw_headers[1:]]

            #Find row with spanwise sensor positions (one line below the line
            # which starts with "# Blade span [m] of stations:")
            if line.startswith("# Blade span [m] of stations:"):
                stations_row = i+1

            # Extract spanwise sensor positions
            if i == stations_row:
                r = np.array(line.split(' ')).astype(float)

            # Once the spanwise stations have been found, look for the first
            # row with numerical values separated by tabspaces or whitespaces
            if r is not None and content_start_row is None \
                and not i == stations_row:
                if re.match(r"^(\d+\.?\d*(\s+|\t))+", line):
                    content_start_row = i
                    break

    # Read data
    data_raw = np.loadtxt(file_path, skiprows=content_start_row)
    times = data_raw[:,0]
    data = data_raw[:,1:]
    data = data.reshape((data.shape[0],len(r),len(headers)), order="F")

    # Save data into a xarray dataset with the spanwise positions and time as
    # coordinates
    data_ds = xr.Dataset(
        {},
        coords={"t":times,
                "r":r}
        )

    for i, header in enumerate(headers):
        data_ds[header] = (list(data_ds.coords.keys()),
                          data[:,:,i])

    return data_ds, times, unit_dict

def check_fpath(file_path):
    """
    Checks if a filepath is the correct data type and if the file exists.

    Args:
        file_path (Union[str | Path]): File path to check.

    Raises:
        TypeError: If fpath is not a string or pathlib Path.
        ValueError: If fpath does not point to a .txt file.
        FileNotFoundError: If the file does not exist.

    Returns:
        fpath (Path): File path as a pathlib.Path object.

    """
    # Check data type
    if isinstance(file_path, str):
        file_path = Path(file_path)
    elif not isinstance(file_path, pathlib.PurePath):
        raise TypeError("File path has to be a string or pathlib Path, "
                        + f"not {type(file_path).__name__}")

    file_path = file_path.resolve()

    #Check if file is a .txt file
    if not file_path.suffix == ".txt":
        raise ValueError(f"File must be a .txt file, not {file_path.suffix}")

    #Check if file exists
    if not file_path.exists():
        raise FileNotFoundError("Input file not found")

    return file_path