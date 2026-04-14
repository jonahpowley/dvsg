import warnings

import numpy as np

from astropy.table import Table, MaskedColumn

from .preprocessing import preprocess_maps_from_plateifu, mask_velocity_maps, mask_binned_map, apply_bin_snr_threshold, apply_sigma_clip

__all__ = [
    "calculate_dvsg",
    "calculate_dvsg_residual",
    "calculate_dvsg_diagnostics",
    "calculate_radial_dvsg",
    "calculate_dvsg_error",
    "calculate_dvsg_from_plateifu",
    "calculate_dvsg_diagnostics_from_plateifu",
    "calculate_radial_dvsg_from_plateifu",
    "return_dvsg_table_from_plateifus",
]


def _load_maps(plateifu, **dvsg_kwargs):
    """Load MaNGA maps only when plateifu routines are called."""

    from .helpers import load_maps

    return load_maps(plateifu, **dvsg_kwargs)


def _load_map_coords(plateifu, **dvsg_kwargs):
    """Load MaNGA map coordinates only when needed."""

    from .helpers import load_map_coords

    return load_map_coords(plateifu, **dvsg_kwargs)


def _return_bin_indices(bin_ids):
    """Load MaNGA bin indices only when needed."""

    from .helpers import return_bin_indices

    return return_bin_indices(bin_ids)


def _return_bin_coord_centres(bin_ra, bin_dec, sv_uindx, gv_uindx):
    """Load MaNGA bin centres only when needed."""

    from .helpers import return_bin_coord_centres

    return return_bin_coord_centres(bin_ra, bin_dec, sv_uindx, gv_uindx)

# ---------------------
# Calculation functions
# ---------------------
def calculate_dvsg(sv_norm, gv_norm):
    """Calculate DVSG from normalised stellar and gas arrays.

    Parameters
    ----------
    sv_norm : array_like
        Normalised stellar velocity map
    gv_norm : array_like
        Normalised gas velocity map

    Returns
    -------
    dvsg : float
        The DVSG value of a galaxy
    """

    if not np.shape(sv_norm) == np.shape(gv_norm):
        raise Exception('sv_norm and gv_norm must have the same shape. Currently have shapes ' + str(np.shape(sv_norm)) + ' and ' + str(np.shape(gv_norm)))

    residual = np.abs(sv_norm - gv_norm)
    n_valid = np.count_nonzero(np.isfinite(residual))
    if n_valid == 0:
        dvsg = np.nan
    else:
        dvsg = np.nanmean(residual)

    return dvsg


def calculate_dvsg_residual(sv_norm, gv_norm):
    """Calculate DVSG residual from normalised stellar and gas arrays.

    Parameters
    ----------
    sv_norm : array_like
        Normalised stellar velocity map
    gv_norm : array_like
        Normalised gas velocity map

    Returns
    -------
    residual : np.ndarray
        The residual value per bin.

    """

    if not np.shape(sv_norm) == np.shape(gv_norm):
        raise Exception('sv_norm and gv_norm must have the same shape. Currently have shapes ' + str(np.shape(sv_norm)) + ' and ' + str(np.shape(gv_norm)))

    residual = np.abs(sv_norm - gv_norm)

    return residual


def calculate_dvsg_diagnostics(sv_norm, gv_norm):
    """Calculate DVSG diagnostics from normalised stellar and gas arrays.

    Parameters
    ----------
    sv_norm : array_like
        Normalised stellar velocity map
    gv_norm : array_like
        Normalised gas velocity map

    Returns
    -------
    output : dict
        Dictionary containing ``dvsg``, ``dvsg_stderr`` and ``residual``.
    """

    residual = calculate_dvsg_residual(sv_norm, gv_norm)
    n_valid = np.count_nonzero(np.isfinite(residual))
    if n_valid == 0:
        dvsg_stderr = np.nan
    else:
        dvsg_stderr = np.nanstd(residual) / np.sqrt(n_valid)

    output = {}
    output["dvsg"] = calculate_dvsg(sv_norm, gv_norm)
    output["dvsg_stderr"] = dvsg_stderr
    output["residual"] = residual

    return output


def calculate_dvsg_error(sv_ivar, gv_ivar, sv_clip, gv_clip):
    """Calculate analytic DVSG uncertainty from inverse variance maps.

    For the derivation of this formula, see Appendix A of Powley et al. (2026).

    Parameters
    ----------
    sv_ivar : array_like
        Inverse variance of stellar velocity map
    gv_ivar : array_like
        Inverse variance of gas velocity map.
    sv_clip : array_like
        Sigma-clipped stellar velocity map
    gv_clip : array_like
        Sigma-clipped gas velocity map
    Returns
    -------
    dvsg_err : float or None
        Uncertainty from error propagation.
    """

    # Convert ivar to sigma, but only where ivar is positive and finite
    sv_ok = np.isfinite(sv_clip) & np.isfinite(sv_ivar) & (sv_ivar > 0)
    gv_ok = np.isfinite(gv_clip) & np.isfinite(gv_ivar) & (gv_ivar > 0)
    ok = sv_ok & gv_ok

    # Check for null entries
    n = np.count_nonzero(ok)
    if n == 0:
        warnings.warn("DVSG error could not be calculated: no okay points", UserWarning)
        return None

    # Calculate unnormalised ranges
    sv_range = np.nanmax(sv_clip) - np.nanmin(sv_clip)
    gv_range = np.nanmax(gv_clip) - np.nanmin(gv_clip)
    if (not np.isfinite(sv_range)) or (not np.isfinite(gv_range)) or (sv_range == 0) or (gv_range == 0):
        warnings.warn("DVSG error could not be calculated: invalid velocity range", UserWarning)
        return None

    sv_err = np.sqrt(1.0 / sv_ivar[ok])
    gv_err = np.sqrt(1.0 / gv_ivar[ok])

    # Calculate error
    term = ((2.0 / sv_range) * sv_err) ** 2 + ((2.0 / gv_range) * gv_err) ** 2
    dvsg_err = (1.0 / n) * np.sqrt(np.nansum(term))

    return dvsg_err


def calculate_radial_dvsg(bin_centres, residual, sort_ascending: bool, **extras):
    """Return bin radii and residuals, optionally sorted by radius.

    Parameters
    ----------
    bin_centres : array_like
        Bin coordinates as ``(n_bin, 2)``.
    residual : array_like
        Residual value per bin.
    sort_ascending : bool
        If True, sort outputs by radius.

    Returns
    -------
    bin_dists : array_like
        Distance of each bin from map centre.
    residual : array_like
        Input residuals, optionally sorted by radius.
    """

    # Calculate distance from each bin to centre
    bin_dists = np.sqrt(np.sum(bin_centres**2, axis=1))  # (x^2+y^2) summed along co-ordinate axis then sqrted

    if sort_ascending:
        bin_dists, residual = zip(*sorted(zip(bin_dists, residual)))

    return bin_dists, residual


# TODO: Add radial DVSG error

# ----------------------
# Routines from plateifu
# ----------------------

def calculate_dvsg_from_plateifu(plateifu, **dvsg_kwargs):
    """Calculate DVSG for one plateifu using pipeline kwargs."""

    # Load preprocessed maps
    sv_norm, gv_norm = preprocess_maps_from_plateifu(plateifu, **dvsg_kwargs)

    # Calculate DVSG
    dvsg = calculate_dvsg(sv_norm, gv_norm)

    return dvsg


def calculate_dvsg_diagnostics_from_plateifu(plateifu, **dvsg_kwargs):
    """Calculate DVSG diagnostics for one plateifu.

    Parameters
    ----------
    plateifu : str
        MaNGA plate-IFU identifier.
    **dvsg_kwargs
        Pipeline options passed through preprocessing and error routines.

    Returns
    -------
    dict
        Dictionary containing ``dvsg`` and ``dvsg_err`` with optional
        ``residual`` when ``return_residual=True``.
    """
    # Load preprocessed maps
    sv_norm, gv_norm = preprocess_maps_from_plateifu(plateifu, **dvsg_kwargs)

    # Load flattened variance and sigma-clipped maps
    sv_map, gv_map, sv_mask, gv_mask, sv_ivar, gv_ivar, bin_ids, bin_snr = _load_maps(plateifu, **dvsg_kwargs)
    sv_flat, gv_flat = mask_velocity_maps(sv_map, gv_map, sv_mask, gv_mask, bin_ids)
    sv_ivar_flat, gv_ivar_flat = mask_velocity_maps(sv_ivar, gv_ivar, sv_mask, gv_mask, bin_ids)
    bin_snr_flat = mask_binned_map(bin_snr, sv_mask, bin_ids)
    if dvsg_kwargs.get("snr_threshold") is not None:
        sv_flat, gv_flat = apply_bin_snr_threshold(sv_flat, gv_flat, bin_snr_flat, **dvsg_kwargs)
    sv_clip, gv_clip = apply_sigma_clip(sv_flat, gv_flat, **dvsg_kwargs)

    output = {}
    output["dvsg"] = calculate_dvsg(sv_norm, gv_norm)
    output["dvsg_err"] = calculate_dvsg_error(sv_ivar_flat, gv_ivar_flat, sv_clip, gv_clip)
    if dvsg_kwargs.get("return_residual", False):
        output["residual"] = calculate_dvsg_residual(sv_norm, gv_norm)

    return output


def calculate_radial_dvsg_from_plateifu(plateifu: str, dvsg_kwargs: dict):
    """Calculate radial DVSG components for one plateifu."""

    dvsg_kwargs = dvsg_kwargs.copy()
    dvsg_kwargs["return_residual"] = True

    output = calculate_dvsg_diagnostics_from_plateifu(plateifu, **dvsg_kwargs)
    residual = output["residual"]

    # Load bin information
    _, _, _, _, _, _, bin_ids, _ = _load_maps(plateifu, **dvsg_kwargs)
    _, _, bin_ra, bin_dec = _load_map_coords(plateifu, **dvsg_kwargs)
    _, sv_uindx, _, gv_uindx = _return_bin_indices(bin_ids)
    bin_centres = _return_bin_coord_centres(bin_ra, bin_dec, sv_uindx, gv_uindx)

    # Calculate radial dvsg
    bin_dists, residual = calculate_radial_dvsg(bin_centres, residual, **dvsg_kwargs)

    return bin_dists, residual


def return_dvsg_table_from_plateifus(plateifus, **dvsg_kwargs):
    """Return a table of DVSG values for a list of plateifus.

    Parameters
    ----------
    plateifus : list[str]
        MaNGA plate-IFU identifiers.
    **dvsg_kwargs
        Keyword arguments forwarded to the DVSG pipeline.

    Returns
    -------
    astropy.table.Table
        Table with ``plateifu``, ``dvsg`` and masked ``dvsg_err`` columns.
    """
    # Create lists to store DVSG data
    dvsgs = []
    dvsg_errs = []
    dvsg_err_masks = []

    # Loop over each plateifu
    for i, plateifu in enumerate(plateifus):

        output = calculate_dvsg_diagnostics_from_plateifu(plateifu, **dvsg_kwargs)

        dvsg = output["dvsg"]
        dvsgs.append(dvsg)

        # append un/masked error values
        dvsg_err = output["dvsg_err"]
        if dvsg_err is None:
            dvsg_errs.append(0.0)
            dvsg_err_masks.append(True)
        else:
            dvsg_errs.append(dvsg_err)
            dvsg_err_masks.append(False)

    # Make astropy Table
    tb = Table()
    tb["plateifu"] = plateifus
    tb["dvsg"] = dvsgs
    tb["dvsg_err"] = MaskedColumn(
        data=dvsg_errs,
        mask=dvsg_err_masks,
        dtype=float
    )

    return tb
