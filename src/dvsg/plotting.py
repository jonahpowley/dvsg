import numpy as np

from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator

from mangadap.util.fitsutil import DAPFitsUtil

from .helpers import load_maps
from .preprocessing import (
    preprocess_maps_from_plateifu,
    mask_velocity_maps, 
    mask_binned_map, 
    apply_bin_snr_threshold, 
    apply_sigma_clip, 
)

__all__ = [
    "transform_flat_to_map",
    "reconstruct_stellar_gas_residual_maps",
    "return_ticks_for_plotting",
    "format_ticks",
    "mask_maps_for_plotting",
    "plot_stellar_gas_residual_maps",
    "plot_stellar_gas_residual_visual_maps",
]

# ----------------------------
# Map reconstruction functions
# ----------------------------

def transform_flat_to_map(flat, map_shape, bins, mask):
    """Reconstruct a 2D map from flattened binned values.

    Parameters
    ----------
    flat : array_like
        1-D array of binned values. NaNs are allowed.
    map_shape : tuple[int, int]
        Target 2D shape (ny, nx) of the reconstructed map.
    bins : array_like
        Integer bin index array used by DAPFitsUtil.reconstruct_map.
    mask : array_like
        Boolean mask or integer mask of shape ``map_shape``. ``True`` entries
        indicate pixels to be masked.

    Returns
    -------
    map_recon : numpy.ndarray
        2D array shaped ``map_shape`` containing the reconstructed map. 
        Masked pixels will be ``np.nan``.
    """

    # Set NaNs to zero
    nan_mask = np.isnan(flat)
    flat[nan_mask] = 0.0

    # Reconstruct maps
    map_recon = DAPFitsUtil.reconstruct_map(map_shape, bins.flatten(), flat)

    # Reapply masks
    map_recon[map_recon == 0] = np.nan
    map_recon[mask.astype(bool)] = np.nan

    return map_recon


def reconstruct_stellar_gas_residual_maps(plateifu: str, **dvsg_kwargs):
    """Reconstruct stellar, gas and residual 2D maps for plotting.

    Parameters
    ----------
    plateifu : str
        MaNGA plate-IFU identifier.
    **dvsg_kwargs :
        Keyword arguments forwarded to ``load_maps`` and preprocessing functions.

    Returns
    -------
    sv_norm_recon, gv_norm_recon, residual_recon : tuple[numpy.ndarray, ...]
        Reconstructed 2D velocity map products (``ny, nx``). Invalid pixels are ``np.nan``.
    """

    # Load bin information
    sv_map, _, sv_mask, _, _, _, bin_ids, bin_snr = load_maps(plateifu, **dvsg_kwargs)
    if dvsg_kwargs.get("snr_threshold") is None:
        bin_snr_mask = np.zeros_like(bin_snr, dtype=bool)
    else:
        bin_snr_mask = bin_snr < dvsg_kwargs["snr_threshold"]

    # Load flat maps
    sv_norm, gv_norm = preprocess_maps_from_plateifu(plateifu, **dvsg_kwargs)

    # Calculate residual map
    residual = np.abs(sv_norm - gv_norm)

    # Reconstruct maps
    map_shape = sv_map.shape
    bins = bin_ids[1]
    mask = sv_mask.astype(bool) | bin_snr_mask
    sv_norm_recon = transform_flat_to_map(sv_norm, map_shape, bins, mask)
    gv_norm_recon = transform_flat_to_map(gv_norm, map_shape, bins, mask)
    residual_recon = transform_flat_to_map(residual, map_shape, bins, mask)

    return sv_norm_recon, gv_norm_recon, residual_recon

# --------------------
# Formatting functions
# --------------------

def return_ticks_for_plotting(plateifu, nticks, dvsg_kwargs):
    """Compute tick locations and (unnormalised) values for plotting.

    Parameters
    ----------
    plateifu : str
        MaNGA plate-IFU identifier.
    nticks : int
        Number of tick marks to produce.
    dvsg_kwargs : dict
        Pipeline options forwarded to preprocessing (e.g. snr_threshold).

    Returns
    -------
    sv_ticks, gv_ticks : tuple[numpy.ndarray, numpy.ndarray]
        unnormalised tick values for stellar and gas velocity ranges respectively.
    """

    # Load map
    sv_map, gv_map, sv_mask, gv_mask, _, _, bin_ids, bin_snr = load_maps(plateifu, **dvsg_kwargs)

    # Extract masked values and flatten
    sv_flat, gv_flat = mask_velocity_maps(sv_map, gv_map, sv_mask, gv_mask, bin_ids)
    bin_snr_flat = mask_binned_map(bin_snr, sv_mask, bin_ids)  # use stellar velocity mask

    # Apply SNR threshold
    if dvsg_kwargs.get("snr_threshold") is not None:
        sv_flat, gv_flat = apply_bin_snr_threshold(sv_flat, gv_flat, bin_snr_flat, **dvsg_kwargs)

    # Sigma clip and normalise maps
    sv_clip, gv_clip = apply_sigma_clip(sv_flat, gv_flat, **dvsg_kwargs)

    sv_ticks = np.linspace(np.nanmin(sv_clip.compressed()), np.nanmax(sv_clip.compressed()), nticks)
    gv_ticks = np.linspace(np.nanmin(gv_clip.compressed()), np.nanmax(gv_clip.compressed()), nticks)

    return sv_ticks, gv_ticks


def format_ticks(sv_ticks, gv_ticks, orig_ticks):
    """Format tick positions and labels using matplotlib FixedLocator and ticklabel lists.

    Parameters
    ----------
    sv_ticks, gv_ticks : array_like
        1D arrays of unnormalised tick values
    orig_ticks : bool
        If True, tick labels include both normalised and original values
        (``norm (orig)``). If False, labels contain only the normalised value.

    Returns
    -------
    sv_ticker, gv_ticker : tuple
        Each entry is a tuple (FixedLocator, list[str]) to be passed to
        colorbar.set_ticks and colorbar.set_ticklabels.
    """

    nticks = len(sv_ticks)
    ticks = np.linspace(-1, 1, nticks)

    # Create stellar and gas velocity ticks
    sv_labels = []
    gv_labels = []
    for nt, st, gt in zip(ticks, sv_ticks, gv_ticks):
        if orig_ticks:
            # Unnormalised
            sv_labels.append(r"{:.2f} ({:.1f})".format(nt, st))
            gv_labels.append(r"{:.2f} ({:.1f})".format(nt, gt))
        else:
            # Normalised
            sv_labels.append(r"{:.2f}".format(nt))
            gv_labels.append(r"{:.2f}".format(nt))

    # Return as FixedLocators
    sv_ticker = FixedLocator(ticks), sv_labels
    gv_ticker = FixedLocator(ticks), gv_labels

    return sv_ticker, gv_ticker


def mask_maps_for_plotting(sv_map, gv_map, sv_mask, gv_mask):
    """Apply boolean masks to stellar and gas maps.

    Parameters
    ----------
    sv_map, gv_map : array_like
        Stellar and gas velocity maps (ny, nx)
    sv_mask, gv_mask : array_like
        Masks for the stellar and gas velocity maps. 
        ``True`` marks invalid/masked pixels (set to ``np.nan``).

    Returns
    -------
    sv_ma, gv_ma : tuple[numpy.ndarray, numpy.ndarray]
        Masked stellar and gas velocity maps
    """
    sv_ma = sv_map.copy()
    gv_ma = gv_map.copy()

    sv_ma[sv_mask.astype(bool)] = np.nan
    gv_ma[gv_mask.astype(bool)] = np.nan

    return sv_ma, gv_ma

# ------------------
# Plotting functions
# ------------------

def plot_stellar_gas_residual_maps(
    ax,
    plateifu,
    x_as, y_as,
    bin_ra, bin_dec,
    sv_norm, gv_norm, residual,
    dvsg,
    dvsg_kwargs: dict,
    dvsg_err = None,
    r_eff: float = None,
    plot_kwargs: dict = None,
):
    """Plot stellar, gas and residual maps on existing axes."""

    # Set plotting arguments
    plot_defaults = {
        "labsize": 20,
        "txtsize": 20,
        "tcksize": 20,
        "labelpad": 0,
        "nticks": 5,
        "orig_ticks": True,
        "plot_bins": False,
        "plot_error": True,
    }
    if plot_kwargs is not None:
        for key in plot_defaults:
            if key not in plot_kwargs:
                plot_kwargs[key] = plot_defaults[key]
    else:
        plot_kwargs = plot_defaults
    labsize = plot_kwargs.get("labsize")
    txtsize = plot_kwargs.get("txtsize")
    tcksize = plot_kwargs.get("tcksize")
    labelpad = plot_kwargs.get("labelpad")
    nticks = plot_kwargs.get("nticks")
    orig_ticks = plot_kwargs.get("orig_ticks")
    plot_bins = plot_kwargs.get("plot_bins")
    plot_error = plot_kwargs.get("plot_error")

    # Create ticks and labels
    sv_ticks, gv_ticks = return_ticks_for_plotting(plateifu, nticks, dvsg_kwargs)
    sv_ticker, gv_ticker = format_ticks(sv_ticks, gv_ticks, orig_ticks)
    sv_loc, sv_labs = sv_ticker
    gv_loc, gv_labs = gv_ticker
    if orig_ticks:
        sv_cbar_label = r"$V_{\star}~\rm{[Norm.~(km~s^{-1})]}$"
        gv_cbar_label = r"$V_{\rm g}~\rm{[Norm.~(km~s^{-1})]}$"
    else:
        sv_cbar_label = r"$V_{\star}~\rm{[Norm.]}$"
        gv_cbar_label = r"$V_{\rm g}~\rm{[Norm.]}$"

    # Stellar panel
    im0 = ax[0].imshow(sv_norm, cmap="RdBu_r", origin="lower", extent=[x_as.max(), x_as.min(), y_as.min(), y_as.max()])
    cb0 = ax[0].figure.colorbar(im0, ax=ax[0], fraction=0.05, pad=0.03)
    cb0.set_label(sv_cbar_label, labelpad=labelpad, fontsize=labsize)
    cb0.set_ticks(sv_loc.locs)
    cb0.set_ticklabels(sv_labs)
    cb0.ax.tick_params(axis="y", which="major", labelsize=tcksize)
    ax[0].text(0.03, 0.97, plateifu, fontsize=txtsize, transform=ax[0].transAxes, va="top", ha="left")

    # Gas panel
    im1 = ax[1].imshow(gv_norm, cmap="RdBu_r", origin="lower", extent=[x_as.max(), x_as.min(), y_as.min(), y_as.max()])
    cb1 = ax[1].figure.colorbar(im1, ax=ax[1], fraction=0.05, pad=0.03)
    cb1.set_label(gv_cbar_label, labelpad=labelpad, fontsize=labsize)
    cb1.set_ticks(gv_loc.locs)
    cb1.set_ticklabels(gv_labs)
    cb1.ax.tick_params(axis="y", which="major", labelsize=tcksize)

    # Residual panel
    im2 = ax[2].imshow(residual, cmap="viridis", origin="lower", extent=[x_as.max(), x_as.min(), y_as.min(), y_as.max()])
    cb2 = ax[2].figure.colorbar(im2, ax=ax[2], fraction=0.05, pad=0.03)
    cb2.set_label(r"Residual [Norm.]", labelpad=labelpad, fontsize=labsize)
    cb2.ax.tick_params(axis="y", which="major", labelsize=tcksize)
    if (dvsg_err is not None) and plot_error:
        dvsg_str = rf"$\Delta V_{{\star-g}}$ = {dvsg:.2f} ± {dvsg_err:.2f}"
    else:
        dvsg_str = rf"$\Delta V_{{\star-g}}$ = {dvsg:.2f}"
    ax[2].text(0.97, 0.01, dvsg_str, fontsize=txtsize, transform=ax[2].transAxes, va="bottom", ha="right")

    # Subplot formatting
    for i in range(3):

        if plot_bins:
            ax[i].scatter(bin_ra, bin_dec, color="k", marker=".", s=50, lw=0)
        if r_eff is not None:
            ax[i].add_patch(Circle((0, 0), r_eff, fill=False, edgecolor="k", linewidth=1.2, transform=ax[i].transData))
        
        ax[i].set_xlabel(r"$\Delta \alpha~[\rm{arcsec}]$", size=labsize)
        ax[i].set_ylabel(r"$\Delta \delta ~[\rm{arcsec}]$", size=labsize)
        ax[i].tick_params(axis="both", which="major", labelsize=tcksize)
        ax[i].set_aspect("equal")

    return ax

def plot_stellar_gas_residual_visual_maps(
    ax,
    plateifu,
    x_as, y_as,
    bin_ra, bin_dec,
    sv_norm, gv_norm, residual, image,
    dvsg,
    dvsg_kwargs: dict,
    dvsg_err = None,
    r_eff: float = None,
    plot_kwargs: dict = None,
):
    """Plot stellar/gas/residual maps plus visual image on existing axes."""

    # Set plotting arguments
    plot_defaults = {
        "labsize": 20,
        "txtsize": 20,
        "tcksize": 20,
        "labelpad": 0,
        "nticks": 5,
        "orig_ticks": True,
        "plot_bins": False,
        "plot_error": True
    }
    if plot_kwargs is not None:
        for key in plot_defaults:
            if key not in plot_kwargs:
                plot_kwargs[key] = plot_defaults[key]
    else:
        plot_kwargs = plot_defaults
    labsize = plot_kwargs.get("labsize")
    txtsize = plot_kwargs.get("txtsize")
    tcksize = plot_kwargs.get("tcksize")
    labelpad = plot_kwargs.get("labelpad")
    nticks = plot_kwargs.get("nticks")
    orig_ticks = plot_kwargs.get("orig_ticks")
    plot_bins = plot_kwargs.get("plot_bins")
    plot_error = plot_kwargs.get("plot_error")

    # Create ticks and labels
    sv_ticks, gv_ticks = return_ticks_for_plotting(plateifu, nticks, dvsg_kwargs)
    sv_ticker, gv_ticker = format_ticks(sv_ticks, gv_ticks, orig_ticks)
    sv_loc, sv_labs = sv_ticker
    gv_loc, gv_labs = gv_ticker
    if orig_ticks:
        sv_cbar_label = r"$V_{\star}~\rm{[Norm.~(km~s^{-1})]}$"
        gv_cbar_label = r"$V_{\rm{g}}~\rm{[Norm.~(km~s^{-1})]}$"
    else:
        sv_cbar_label = r"$V_{\star}~\rm{[Norm.]}$"
        gv_cbar_label = r"$V_{\rm{g}}~\rm{[Norm.]}$"

    # Stellar panel
    im0 = ax[0].imshow(sv_norm, cmap="RdBu_r", origin="lower", extent=[x_as.max(), x_as.min(), y_as.min(), y_as.max()])
    cb0 = ax[0].figure.colorbar(im0, ax=ax[0], fraction=0.0465, pad=0.03)
    cb0.set_label(sv_cbar_label, labelpad=labelpad, fontsize=labsize)
    cb0.set_ticks(sv_loc.locs)
    cb0.set_ticklabels(sv_labs)
    cb0.ax.tick_params(axis="y", which="major", labelsize=tcksize)
    ax[0].text(0.03, 0.97, plateifu, fontsize=txtsize, transform=ax[0].transAxes, va="top", ha="left")

    # Gas panel
    im1 = ax[1].imshow(gv_norm, cmap="RdBu_r", origin="lower", extent=[x_as.max(), x_as.min(), y_as.min(), y_as.max()])
    cb1 = ax[1].figure.colorbar(im1, ax=ax[1], fraction=0.0465, pad=0.03)
    cb1.set_label(gv_cbar_label, labelpad=labelpad, fontsize=labsize)
    cb1.set_ticks(gv_loc.locs)
    cb1.set_ticklabels(gv_labs)
    cb1.ax.tick_params(axis="y", which="major", labelsize=tcksize)

    # Residual panel
    im2 = ax[2].imshow(residual, cmap="viridis", origin="lower", extent=[x_as.max(), x_as.min(), y_as.min(), y_as.max()])
    cb2 = ax[2].figure.colorbar(im2, ax=ax[2], fraction=0.0465, pad=0.03)
    cb2.set_label(r"Residual [Norm.]", labelpad=labelpad, fontsize=labsize)
    cb2.ax.tick_params(axis="y", which="major", labelsize=tcksize)
    if (dvsg_err is not None) and plot_error:
        dvsg_str = rf"$\Delta V_{{\star-g}}$ = {dvsg:.2f} ± {dvsg_err:.2f}"
    else:
        dvsg_str = rf"$\Delta V_{{\star-g}}$ = {dvsg:.2f}"
    ax[2].text(0.97, 0.00, dvsg_str, fontsize=txtsize, transform=ax[2].transAxes, va="bottom", ha="right")

    # Visual panel
    # if isinstance(image, Image.Image):
    #     image = np.asarray(image.data)
    # else:
    #     image = np.asarray(image)
    image = np.asarray(image)
    im3 = ax[3].imshow(image, origin="upper")

    # Subplot formatting
    for i in range(4):
        ax[i].set_aspect("equal")

        # DVSG panels
        if i < 3:
            if plot_bins:
                ax[i].scatter(bin_ra, bin_dec, color="k", marker=".", s=50, lw=0)
            if r_eff is not None:
                ax[i].add_patch(Circle((0, 0), r_eff, fill=False, edgecolor="k", linewidth=1.2, transform=ax[i].transData))
            ax[i].set_xlabel(r"$\Delta \alpha \ \;[\mathrm{arcsec}]$", size=labsize)
            ax[i].set_ylabel(r"$\Delta \delta \ \;[\mathrm{arcsec}]$", size=labsize)

        # Image panel
        elif i == 3:
            ax[i].set_xticks([])
            ax[i].set_yticks([])

    return ax
