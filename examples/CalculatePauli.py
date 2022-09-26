#This is an example on how to calculate the pauli vector for an fsar scene.
#It includes the use of dask to spread the calculation on multiple cores.
#With dask, you could also process datasets bigger than the RAM of your computer. The key idea is that the data can be splitted along azimuth and range, as long as all calculations are indepentend for every pixel.
# (A counterexample would be a multilook operation in azimuth and range. However, even this can easily be parallelized, see the multilook implementation in sartools.py)
#In this case, since the calculation of the Pauli vector is just a simple addition, dask is not necessary, but the same workflow can be applied to much more complex calculations.
#
#Necessary packages: xarray, dask

import xarray as xr
from sartools.sartools import  create_pauli3
from sartools.helpers import convert_complex, convert_real
if __name__=="__main__":
    from dask.distributed import Client, LocalCluster
    datadir="/path/to/data/slc_15arctic0808_L*.nc" #read in multiple .nc files, one for each polarization
    savedir="pauli.nc"

    cluster=LocalCluster(n_workers=4, threads_per_worker=6, memory_limit='20GB')
    client=Client(cluster)
    print(client)

    slc=xr.open_mfdataset(str(datadir), chunks={"az":2000, "rg":2000}).slc
    slc=convert_complex(slc)
    #Optional: for testing, select a subset of the data along azimuth
    # selection={"az":slice(0,4000)}
    # slc=slc.isel(selection)
    pauli=create_pauli3(slc)
    pauli=convert_real(pauli)
    pauli.name="pauli_vector"

    pauli.to_netcdf(savedir)
    client.close()
    cluster.close()