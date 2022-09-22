import numpy as np
def create_rgb(d, dim, coords=None):
    """Create RGB values from a range-azimuth DataArray.

    The absolute values along dimension "dim" are taken as RGB values. They are clipped to 2.5 times the rg-az mean and normalized to (0,255).

    Args:
        d (DataArray with dimensions "rg","az"): The data.
        dim (str): Dimension with length 3 to interpret as RGB
        coords (dict): Optional mapping from coordinate labels along "dim" to RGB coordinate labels. Defaults to None, in which case the output will have coordinate labels 'r', 'g' and 'b'.

    Returns:
        DataArray: DataArray of type uint8 with dimension 'rgb'. Can be directly used in matplotlib.imshow.
    """
    assert(len(d[dim]==3))
    rgb=['r', 'g', 'b']
    if coords: 
        rgb=[coords[c] for c in d[dim].values]
    d=d.rename({dim:'rgb'})
    d.coords["rgb"]=('rgb', rgb)
    d=abs(d)#Take absolute
    d=d.clip(0,2.5*d.mean(dim=["rg","az"]))#limit range
    d=d/d.max(dim=["rg","az"])*255#normalize
    d=d.astype(np.uint8)
    return d.sel(rgb=['r', 'g', 'b']).transpose(...,'rg', 'az','rgb')