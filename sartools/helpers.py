import numpy as np
import xarray as xr
import pathlib
import struct

def collapse_chunks(data, dims):
    """Collapse all dask chunks along the given dimensions

    Args:
        data (chunked DataArray): DataArray to rechunk
        dims (array): List of dimensions

    Returns:
        DataArray: DataArray with only one chunk along each specified dimension
    """
    return data.chunk({d:data.sizes[d] for d in dims})
def convert_real(ds):
    ds = xr.concat([ds.real, ds.imag], dim='complex')
    ds.coords["complex"]=("complex", ["re", "im"])
    return ds
def convert_complex(dataset):
    return dataset.isel(complex=0) + 1j * dataset.isel(complex=1)

def opendat(input_file, tot_kb, complex=True):
    if isinstance(input_file, str):
        input_file = pathlib.Path(input_file)
    with open(input_file, 'rb') as f:
        
        nbytes = 4 
        
        # Dimensions are in the first 8 bytes of the file's header  
        
        nrg = int.from_bytes(f.read(nbytes), 'big')    
        f.seek(nbytes) # Cursor after 4 bytes: second dimension
        naz = int.from_bytes(f.read(nbytes), 'big')    
        print(f'Range: {nrg}, Azimuth: {naz}')
                
        # f.seek(i) # Cursor after 8 bytes: data in complex format, real part    
        # data_r = struct.unpack('>f', f.read(nbytes))[0]    
        # f.seek(i+ibytes) # Cursor after 12 bytes: data in complex format, imaginary part
        # data_i = struct.unpack('>f', f.read(nbytes))[0]    
        # data_c = data_r + 1j*data_i
            
        byte_step = 4
        ibytes = 4
        ii = 0
        data = []
        
        
        while ibytes <= (tot_kb-8): # substract 8 because it reads 4 bytes at once
            f.seek(ibytes+byte_step)
            data.append(struct.unpack('>f', f.read(nbytes))[0])
            ii = ii + 1
            ibytes = ibytes + byte_step

    data_aux = np.array(data)
    if complex:
        real_part = data_aux[::2].reshape((nrg,naz), order='F')
        imag_part = data_aux[1::2].reshape((nrg,naz), order='F')
        data = real_part + 1j*imag_part
    else:
        data=data_aux.reshape((nrg,naz), order='F')
    data=xr.DataArray(data, coords=[('rg', np.arange(data.shape[0])),('az', np.arange(data.shape[1]))])
    data.name=input_file.stem
    return data