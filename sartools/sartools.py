import numpy as np
from scipy.ndimage import uniform_filter
import xarray as xr

def phase(data):
    phase= np.arctan2(data.imag, data.real)
    phase=phase.where(abs(data)>1e-14, other=0.0)
    return phase

def multilook(data, filtersize={"rg":5,"az":5}, moving=True):
    """Multilook by averaging the data along specified dimensions.

    Args:
        data (DataArray): DataArray to multilook
        filtersize (dict, optional): Mapping from dimension names to window size. Defaults to {"rg":5,"az":5}.
        moving (bool, optional): If true, a moving average/convolution will be applied and the size of the output DataArray is unchanged. If False, a blockwise average is applied, reducing the output size. Defaults to True.

    Returns:
        DataArray: Multilooked DataArray
    """
    if moving:
        res=data.rolling(filtersize, center=True).mean()
    else:
        res=data.coarsen(filtersize, boundary='trim').mean()
    return res


def create_pauli(data, poldim="pol"):
    """Create full Pauli vector with components 'hh+vv', 'hh-vv', 'vh+hv', 'vh-hv'.

    Args:
        data (DataArray): Input data
        poldim (str, optional): Name of the dimension which describes polarization. Must have coordinates 'hh', 'vv', 'hv', 'vh'. Defaults to "pol".

    Returns:
        DataArray: Pauli Vector
    """
    hh=data.sel({poldim:"hh"})
    vv=data.sel({poldim:"vv"})
    hv=data.sel({poldim:"hv"})
    vh=data.sel({poldim:"vh"})
    a=hh+vv
    b=hh-vv
    c=vh+hv
    d=vh-hv
    pauli=xr.concat([a,b,c,d], dim='pauli')/np.sqrt(2)
    pauli.coords["pauli"]=("pauli", ["hh+vv", "hh-vv","vh+hv", "vh-hv"])
    return pauli

def create_pauli3(data, poldim="pol"):
    """Create 3D Pauli vector with components 'hh+vv', 'hh-vv', '2hv'.

    Args:
        data (DataArray): Input data
        poldim (str, optional): Name of the dimension which describes polarization. Must have coordinates 'hh', 'vv', 'hv'. Defaults to "pol".

    Returns:
        DataArray: Pauli Vector
    """
    hh=data.sel({poldim:"hh"})
    vv=data.sel({poldim:"vv"})
    hv=data.sel({poldim:"hv"})
    a=hh+vv
    b=hh-vv
    c=2*hv
    pauli=xr.concat([a,b,c], dim='pauli')/np.sqrt(2)
    pauli.coords["pauli"]=("pauli", ["hh+vv", "hh-vv","2hv"])
    return pauli

def create_lexi(data, poldim="pol"):
    """Create lexicographic vector with components 'hh', 'sqrt(2)*hv', 'vv'.

    Args:
        data (DataArray): Input data
        poldim (str, optional): Name of the dimension which describes polarization. Must have coordinates 'hh', 'vv', 'hv'. Defaults to "pol".

    Returns:
        DataArray: Pauli Vector
    """
    hh=data.sel({poldim:"hh"})
    vv=data.sel({poldim:"vv"})
    hv=data.sel({poldim:"hv"})
    return xr.concat([hh, np.sqrt(2)*hv, vv], dim='lexi')

def calc_corr(data1, data2, norm=False, filtersize={"rg":5,"az":5}, moving=True):
    """Calculate correlation between two DataArrays

    Args:
        data1 (DataArray): First Array
        data2 (DataArray): Second Array
        norm (bool, optional): Normalize by the square-root of the product of the autocorrelations to obtain values between 0 and 1. Defaults to False.
        filtersize (dict, optional): Mapping from dimension names to window size. Defaults to {"rg":5,"az":5}.
        moving (bool, optional): If true, a moving average/convolution will be applied and the size of the output DataArray is unchanged. If False, a blockwise average is applied, reducing the output size. Defaults to True.

    Returns:
        DataArray: Correlation between data1 and data2
    """
    mat=data1*np.conj(data2)
    mat=multilook(mat, filtersize, moving)
    # mat=mat.where(abs(mat)>1e-10)
    if norm:
        auto1=calc_autocorr(data1,norm=False,filtersize=filtersize)
        auto2=calc_autocorr(data2,norm=False,filtersize=filtersize)
        mat=mat/np.sqrt(auto1*auto2)
    return mat

def calc_autocorr(data, filtersize={"rg":5,"az":5}, moving=True):
    """Calculate the autocorrelation of an array.

    Args:
        data (DataArray): Input Array
        filtersize (dict, optional): Mapping from dimension names to window size. Defaults to {"rg":5,"az":5}.
        moving (bool, optional): If true, a moving average/convolution will be applied and the size of the output DataArray is unchanged. If False, a blockwise average is applied, reducing the output size. Defaults to True.

    Returns:
        _type_: _description_
    """
    return calc_corr(data, data, norm=False, filtersize=filtersize, moving=moving)

def calc_corr_matrix(data1, data2, corrdim, norm=False, filtersize={"rg":5,"az":5}, moving=True):
    """Calculate a correlation matrix between two DataArrays by performing an outer product along the specified dimension. The result will contain the correlation between all possible combinations of the elements along 'corrdim'.

    Args:
        data1 (DataArray): First Array
        data2 (DataArray): Second Array
        corrdim (str): Name of the dimension where the outer product is performed. Must be present in data1 and data2 with the same coordinate labels.
        norm (bool, optional): Normalize correlation by the square-root of the product of the autocorrelations to obtain values between 0 and 1. Defaults to False.
        filtersize (dict, optional): Mapping from dimension names to window size. Defaults to {"rg":5,"az":5}.
        moving (bool, optional): If true, a moving average/convolution will be applied and the size of the output DataArray is unchanged. If False, a blockwise average is applied, reducing the output size. Defaults to True.

    Returns:
        DataArray: DataArray with two new dimensions 'co1' and 'co2', each with the coordinates of 'corrdim', representing the different correlation combinations.
    """
    #Idea: xarray will perform an outer product when broadcasting the two datasets with different coordinates, but create a 1D power-vector for the autocorrelation!
    data1_r=data1.rename({corrdim:"co1"})
    data2_r=data2.rename({corrdim:"co2"})
    mat=calc_corr(data1_r, data2_r, norm=norm, filtersize=filtersize, moving=moving)
    return mat
    
def calc_autocorr_matrix(data, corrdim, norm=False, filtersize={"rg":5,"az":5}, moving=True):
    """Calculate a correlation matrix by performing an outer product along the specified dimension. The result will contain the correlation between all possible combinations of the elements along 'corrdim'.

    Args:
        data1 (DataArray): First Array
        data2 (DataArray): Second Array
        corrdim (str): Name of the dimension where the outer product is performed. Must be present in data1 and data2 with the same coordinate labels.
        norm (bool, optional): Normalize correlation by the square-root of the product of the autocorrelations to obtain values between 0 and 1. Defaults to False.
        filtersize (dict, optional): Mapping from dimension names to window size. Defaults to {"rg":5,"az":5}.
        moving (bool, optional): If true, a moving average/convolution will be applied and the size of the output DataArray is unchanged. If False, a blockwise average is applied, reducing the output size. Defaults to True.

    Returns:
        DataArray: DataArray with two new dimensions 'co1' and 'co2', each with the coordinates of 'corrdim', representing the different correlation combinations.
    """
    return calc_corr_matrix(data,data,corrdim, norm=norm, filtersize=filtersize, moving=moving)

# def calc_eigen(data, matdim1, matdim2):
#     "Deprecated. Based on numpy.linalg.eig, sort using xarray indexing"
#     eigenvalues, eigenvectors=xr.apply_ufunc(np.linalg.eig, data, input_core_dims=[[matdim1, matdim2]], output_core_dims=[["eigval"], ["eigval", "index"]])
#     eigenvalues=np.real(eigenvalues)
#     sort_ind=xr.apply_ufunc(lambda d: np.argsort(d)[..., ::-1], eigenvalues, input_core_dims=[["eigval"]], output_core_dims=[["eigval"]])#sort invers by eigenvalue
#     eigenvalues=eigenvalues.isel(eigval=sort_ind)#apply sort to eigenvalues and eigenvectors
#     eigenvectors=eigenvectors.isel(eigval=sort_ind)
#     return eigenvalues, eigenvectors

def calc_eigh(data, matdim1, matdim2):
    """Calculate eigenvalues and eigenvectors for a hermitian matrix.

    NaN values are accepted and lead to NaN in the corresponding eigenvalues and eigenvectors

    Args:
        data (DataArray): Input data. Must be hermitian!
        matdim1 (str): Dimension to be interpreted as row dimension
        matdim2 (str): Dimension to be interpreted as column dimension

    Returns:
        DataArray: Eigenvalues sorted in descending order along dimension 'eigval'
        DataArray: Corresponding eigenvectors, defined along dimension 'index'.
    """
    assert(len(data[matdim1])==len(data[matdim2]))
    output_sizes={"eigval":len(data[matdim1]), "index":len(data[matdim1])}

    def eigh_nan(nparr):
        mask=np.isnan(nparr).any((-2,-1))
        nparr[mask]=0+0j
        val, vec=np.linalg.eigh(nparr) #eigh returns a sorted array!
        val[mask]=np.nan
        vec[mask]=np.nan
        return val[...,::-1], vec[...,::-1] #invert to have biggest value first

    eigenvalues, eigenvectors=xr.apply_ufunc(eigh_nan, data, input_core_dims=[[matdim1, matdim2]], output_core_dims=[["eigval"], ["index", "eigval"]], dask="parallelized", output_dtypes=[float, complex], dask_gufunc_kwargs={"output_sizes":output_sizes}) 
    return eigenvalues, eigenvectors

def calc_eigvalsh(data, matdim1, matdim2):
    """Calculate eigenvalues for a hermitian matrix.
    
    NaN values are accepted and lead to NaN in the corresponding eigenvalues.

    Args:
        data (DataArray): Input data. Must be hermitian!
        matdim1 (str): Dimension to be interpreted as row dimension
        matdim2 (str): Dimension to be interpreted as column dimension

    Returns:
        DataArray: Eigenvalues sorted in descending order along dimension 'eigval'
    """
    """Only for hermitian matrix!"""
    assert(len(data[matdim1])==len(data[matdim2]))
    output_sizes={"eigval":len(data[matdim1])}

    def eigvalsh_nan(nparr):
        mask=np.isnan(nparr).any((-2,-1))
        nparr[mask]=0+0j
        val=np.linalg.eigvalsh(nparr) #eigh returns a sorted array!
        val[mask]=np.nan
        return val[...,::-1] #invert to have biggest value first
    eigenvalues=xr.apply_ufunc(eigvalsh_nan, data, input_core_dims=[[matdim1, matdim2]], output_core_dims=[["eigval"]], dask="parallelized", output_dtypes=[float], dask_gufunc_kwargs={"output_sizes":output_sizes}) 
    return eigenvalues

def calc_eigprob(eigval):
    """Calculate probability values from eigenvalues.

    Args:
        eigval (DataArray): Eigenvalues. Must have one dimension 'eigval'

    Returns:
        DataArray: Probability values
    """
    return eigval/eigval.sum(dim="eigval")

def calc_entropy(eigval):
    """Calculate entropy from eigenvalues

    Args:
        eigval (DataArray): Eigenvalues. Must have one dimension 'eigval'

    Returns:
        DataArray: Entropy values
    """
    P=calc_eigprob(eigval)
    H=-P*np.log(P)/np.log(3)
    H=H.sum(dim="eigval")
    H.name="entropy"
    return H

def calc_anisotropy(eigval):
    """Calculate anisotropy from eigenvalues

    Args:
        eigval (DataArray): Eigenvalues. Must have one dimension 'eigval'

    Returns:
        DataArray: Anisotropy values
    """
    ani=(eigval.isel(eigval=1)-eigval.isel(eigval=2))/(eigval.isel(eigval=1)+eigval.isel(eigval=2))
    ani.name='anisotropy'
    return ani

def calc_mean_apha(eigval, eigvec):
    """Calculate mean alpha angle from eigenvalues

    Args:
        eigval (DataArray): Eigenvalues. Must have one dimension 'eigval'
        eigvec (DataArray): Eigenvectors. Must have two dimensions 'eigval' and 'index'

    Returns:
        DataArray: Anisotropy values in [rad]
    """
    P=calc_eigprob(eigval)
    alpha=np.arccos(abs(eigvec.isel(index=0)))
    alpha=P*alpha
    alpha=alpha.sum(dim="eigval")
    alpha.name='alpha'
    alpha.attrs["long_name"]='mean alpha angle'
    alpha.attrs["units"]='radian'
    return alpha