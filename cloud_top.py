"""Cloud top analysis

Contains functions to determine cloud base and cloud top properties.

This module contains the following functions:
    * cloud_base_top
    * var_below_cloud_top
"""

# Import external packages.
import numpy as np
import xarray as xr


def cloud_base_top(clw, clw_thresh=1.0e-5):
    """Find the model levels corresponding to cloud base and cloud top.

    Parameters
    ----------
    clw : xr.DataArray
        Cloud liquid water field. Must have a model level dimension named
        'lev'. Default units are kg m-3
    clw_thresh : xr.DataArray
        Threshold for detecting presence of liquid water. Default units are
        kg m-3.

    Returns
    -------
    idx_acb : xarray DataArray
        Index of model level directly above cloud base.
    idx_bct : xarray DataArray
        Index of model level directly below cloud top.
    """
    # Determine model level at cloud base and cloud top.
    clw_mask = (  # 3D mask for cloud liquid water.
        xr.where(clw > clw_thresh, True, False)
    )
    clt_mask = (  # 2D mask for presence of clouds.
        xr.where((clw < clw_thresh).all(dim='lev'), False, True)
    )
    lev_acb = (  # First level above cloud base. nan where no clouds.
        clw_mask.idxmax(dim='lev').where(clt_mask)
    )
    idx_acb = (  # Index of first level above cloud base. 0 where no clouds.
        clw_mask.argmax(dim='lev')
    )
    # To get index for above cloud top, make mask that is true everywhere below
    # cloud top (including below cloud base). NOTE: level descends with index,
    # so levi < levj implies levi is above levj!
    bct_mask = (
        xr.where(clw_mask.lev < lev_acb, clw_mask, 0)
        + xr.where(clw_mask.lev >= lev_acb, 1, 0)
    )
    idx_act = bct_mask.argmin(dim='lev')
    # idx_bcb = np.maximum(idx_acb - 1, 0)
    idx_bct = np.maximum(idx_act - 1, 0)

    return idx_acb, idx_bct, clt_mask


def var_below_cloud_top(
        *,
        clt_mask: xr.DataArray,
        idx_bct: xr.DataArray,
        var: xr.DataArray,
        ) -> xr.DataArray:
    """Select variable at model level below cloud top

    Parameters
    ----------
    var : xr.DataArray
        The variable to select below the cloud top. Must include a model level
        dimension called 'lev'.
    idx_bct : xr.DataArray
        The index of the model level below the cloud top. Must have same
        dimensions as var, minus 'lev'.
    clt_mask : xr.DataArray
        Mask of grid cells with liquid water. Must have same dimensions as
        idx_bct.

    Returns
    -------
    var_bct : xarray DataArray
        var selected on the model level below the cloud top.
    """
    var_to_choose = (  # Have to limit levels to 31 in order to use np.choose.
        var.isel(lev=slice(0, 31)).transpose('lev', *idx_bct.dims)
    )
    dims_to_choose = list(var_to_choose.dims)
    dims_to_choose.remove('lev')
    dims_to_choose.insert(0, 'lev')
    print(dims_to_choose)
    var_bct = xr.where(
        clt_mask,
        np.choose(
            idx_bct,
            choices=var_to_choose.transpose(*dims_to_choose),
            mode='clip'),
        np.nan)
    return var_bct
