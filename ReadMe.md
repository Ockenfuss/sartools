# Tools for SAR processing
This repository contains functions to analyze SAR images (mainly polarimetric analysis so far).
Those functions are built to work with xarray DataArrays (https://docs.xarray.dev/en/stable/index.html)

Therefore, it is necessary to define some (short) names for the corresponding dimensions in the DataArrays. Using those names allows the functions to do operations on specific dimensions, like range or polarization.
* Range dimension: "rg"
* Azimuth dimension: "az"
* Polarization: "pol" with labels "hh", "vv", "hv", ...
* Flight track/pass: "track" (the word "Pass" is already a reserved keyword in python, so I decided to use "track")



