import numpy as np

__all__ = [
    "exclude_above_n_sigma",
    "minmax_normalise_velocity_map",
    "mask_velocity_maps",
    "mask_binned_map",
    "apply_bin_snr_threshold",
    "apply_velocity_snr_threshold",
    "apply_sigma_clip",
    "normalise_map",
    "preprocess_maps_from_plateifu",
]


def _return_bin_indices(bin_ids):
    """Load bin indices only when MaNGA helpers are needed."""

    from .helpers import return_bin_indices

    return return_bin_indices(bin_ids)


def _load_maps(plateifu, **dvsg_kwargs):
    """Load MaNGA maps only when plateifu routines are called."""

    from .helpers import load_maps

    return load_maps(plateifu, **dvsg_kwargs)

# ------------------------
# Sigma-clipping functions
# ------------------------

def exclude_above_n_sigma(velocity_map, n: int):
    """Excludes any values in a velocity map greater than n standard deviations
    from the mean velocity of the map (excluding NaNs).

    The standard deviation used is the sample standard deviation (ddof=1).

    Parameters
    ----------
    velocity_map : array_like
        The original velocity map.

    Returns
    -------
    excluded_velocity_map : np.ndarray
        The n-sigma clipped velocity map
    """

    velocity_above_n_sigma = np.nanmean(velocity_map) + n * np.nanstd(velocity_map, ddof=1)
    velocity_below_n_sigma = np.nanmean(velocity_map) - n * np.nanstd(velocity_map, ddof=1)
    
    excluded_velocity_map = velocity_map.copy()

    excluded_velocity_map[velocity_map > velocity_above_n_sigma] = np.nan
    excluded_velocity_map[velocity_map < velocity_below_n_sigma] = np.nan

    return excluded_velocity_map


# -----------------------
# Normalisation functions
# -----------------------
def minmax_normalise_velocity_map(velocity_map):
    """
    Normalises the given velocity map to the range [-1, 1] using the formula:

        x' = 2 * ((x - min(x)) / (max(x) - min(x))) - 1 

    Parameters
    ----------
    velocity_map : array_like
        Input velocity values.

    Returns
    -------
    np.ndarray
        Normalised map. NaNs are preserved.
    """

    velocity_map = np.asarray(velocity_map, dtype=float)  # ensure float copy

    min_val, max_val = np.nanmin(velocity_map), np.nanmax(velocity_map)

    if min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
        return np.full_like(velocity_map, np.nan)

    normalised_velocity_map = 2 * (velocity_map - min_val) / (max_val - min_val) - 1

    return normalised_velocity_map


# -----------------------
# Preprocessing functions
# -----------------------
def mask_velocity_maps(sv_map: np.ndarray, gv_map: np.ndarray, sv_mask: np.ndarray, gv_mask: np.ndarray, bin_ids: np.ndarray, **extras):
    """Flatten stellar/gas maps to one value per bin and apply masks.

    Parameters
    ----------
    sv_map, gv_map : np.ndarray
        Stellar and gas map arrays.
    sv_mask, gv_mask : np.ndarray
        Bitmask arrays for the corresponding maps.
    bin_ids : np.ndarray
        BINID cube used to index bin representatives.

    Returns
    -------
    sv_flat, gv_flat : np.ma.MaskedArray
        Flattened, masked stellar and gas values.
    """

    # Get the unique indices of the stellar and gas velocity bins
    _, sv_uindx, _, gv_uindx = _return_bin_indices(bin_ids)

    # Apply mask and save a masked array
    sv_flat = np.ma.MaskedArray(sv_map.ravel()[sv_uindx], mask=sv_mask.ravel()[sv_uindx] > 0)
    gv_flat = np.ma.MaskedArray(gv_map.ravel()[gv_uindx], mask=gv_mask.ravel()[gv_uindx] > 0)

    return sv_flat, gv_flat


def mask_binned_map(map, mask, bin_ids, **extras):
    """Flatten a binned map to stellar-bin representatives and apply mask."""

    # Get the unique indices of the stellar and gas velocity bins
    _, sv_uindx, _, _ = _return_bin_indices(bin_ids)

    flat = np.ma.MaskedArray(map.ravel()[sv_uindx], mask=mask.ravel()[sv_uindx] > 0)

    return flat


def apply_bin_snr_threshold(sv_flat, gv_flat, bin_snr_flat, snr_threshold: float, **extras):
    """Mask stellar/gas bins below a bin-level SNR threshold."""

    if snr_threshold is None:
        return sv_flat, gv_flat

    reject = bin_snr_flat < snr_threshold

    sv_flat[reject] = np.nan
    gv_flat[reject] = np.nan

    return sv_flat, gv_flat


def apply_velocity_snr_threshold(sv_flat, gv_flat, sv_ivar_flat, gv_ivar_flat, snr_threshold: float, **extras):
    """Mask bins below velocity SNR threshold computed from IVAR."""

    # Calculate S/N ratio of velocity values
    sv_snr = np.abs(sv_flat) / (1 / np.sqrt(sv_ivar_flat))
    gv_snr = np.abs(gv_flat) / (1 / np.sqrt(gv_ivar_flat))

    # Apply mask
    sv_reject = sv_snr < snr_threshold
    gv_reject = gv_snr < snr_threshold

    sv_flat[sv_reject] = np.nan
    gv_flat[gv_reject] = np.nan

    return sv_flat, gv_flat


def apply_sigma_clip(sv_flat, gv_flat, n_sigma: float, **extras):
    """Apply symmetric n-sigma clipping to stellar and gas arrays."""

    # Apply sigma clip
    sv_excl = exclude_above_n_sigma(sv_flat, n_sigma)
    gv_excl = exclude_above_n_sigma(gv_flat, n_sigma)

    return sv_excl, gv_excl


def normalise_map(sv_excl, gv_excl, **extras):
    """Normalise preprocessed stellar and gas arrays using min-max scaling.

    Parameters
    ----------
    sv_excl : array_like
        Preprocessed stellar velocity map values.
    gv_excl : array_like
        Preprocessed gas velocity map values.
    Returns
    -------
    sv_norm, gv_norm : np.ndarray
        Normalised stellar and gas velocity maps.
    """

    sv_norm = minmax_normalise_velocity_map(sv_excl)
    gv_norm = minmax_normalise_velocity_map(gv_excl)

    return sv_norm, gv_norm


# -----------------------------------
# Multi-stage preprocessing functions
# -----------------------------------
def preprocess_maps_from_plateifu(plateifu: str, **dvsg_kwargs):
    """Run the standard preprocessing chain for one plateifu.

    Steps are: load maps, flatten/mask bins, optional bin-SNR cut,
    sigma clipping, then map normalisation.
    """
    if "norm_method" in dvsg_kwargs:
        raise TypeError(
            "norm_method is no longer supported in the public DVSG pipeline. "
            "The pipeline now always uses min-max normalisation."
        )

    # Load map
    sv_map, gv_map, sv_mask, gv_mask, _, _, bin_ids, bin_snr = _load_maps(plateifu, **dvsg_kwargs)
    # Extract masked values and flatten
    sv_flat, gv_flat = mask_velocity_maps(sv_map, gv_map, sv_mask, gv_mask, bin_ids)
    bin_snr_flat = mask_binned_map(bin_snr, sv_mask, bin_ids)  # use stellar velocity mask

    # Apply SNR threshold
    if dvsg_kwargs.get("snr_threshold") is not None:
        sv_flat, gv_flat = apply_bin_snr_threshold(sv_flat, gv_flat, bin_snr_flat, **dvsg_kwargs)

    # Sigma clip and normalise maps
    sv_clip, gv_clip = apply_sigma_clip(sv_flat, gv_flat, **dvsg_kwargs)
    sv_norm, gv_norm = normalise_map(sv_clip, gv_clip, **dvsg_kwargs)

    return sv_norm, gv_norm
