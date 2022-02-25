"""
Functions for computing binned responses.
"""

from itertools import product

import numpy as np

from roto import circstats

from ..lib.motion import NORM_POS_RANGE, MDIR_RANGE, CIRCLE_DIAMETER
from ..lib.lfp import PHASE_MIN, PHASE_MAX


DEFAULT_MIN_OCC = 1.0
DEFAULT_BINS = 15
INFO_MIN_OCC = 0.5
INFO_BINS = DEFAULT_BINS

DEFAULT_DIR_BINS = 36
INFO_DIR_BINS = DEFAULT_DIR_BINS

DEFAULT_SPEED_MIN_OCC = 8.0
DEFAULT_SPEED_BINS = 14
DEFAULT_SPEED_MIN = 5.0  # cm/s
DEFAULT_SPEED_MAX = 40.0  # cm/s
INFO_SPEED_MIN_OCC = 3.0
INFO_SPEED_BINS = 18
INFO_SPEED_MIN = DEFAULT_SPEED_MIN
INFO_SPEED_MAX = 50.0

DEFAULT_PHASE_MIN_SPIKES = 5
DEFAULT_PHASE_BINS = 36
INFO_PHASE_BINS = DEFAULT_PHASE_BINS
INFO_PHASE_POS_BINS = INFO_BINS
INFO_PHASE_POS_XRANGE = NORM_POS_RANGE[0]
INFO_PHASE_POS_YRANGE = NORM_POS_RANGE[1]


_norm = lambda H: H / H.sum()

def skaggs(spkmap, occmap, valid):
    """Compute the Skaggs spatial information rate for a spikemap.

    Arguments:
    spkmap -- binned spike-count map
    occmap -- binned occupancy map
    valid -- binned boolean map or index array of valid pixels

    Returns:
    float, spike information rate (bits/spike)
    """
    spk = spkmap[valid]
    occ = occmap[valid]

    if occ.sum() == 0.0:
        return 0.0

    F = spk.sum() / occ.sum()
    if F == 0.0:
        return 0.0

    p = occ / occ.sum()
    f = spk / occ
    nz = np.nonzero(f)

    I = np.sum(p[nz] * f[nz] * np.log2(f[nz] / F)) / F

    return I

def rateinfo(x, y, Xp, Yp):
    """Compute the spatial information rate for spike rate."""
    spk = spikemap(x, y, bins=INFO_BINS)
    occ = occmap(Xp, Yp, bins=INFO_BINS, min_occ=INFO_MIN_OCC)
    valid = np.logical_not(np.isnan(occ))
    return skaggs(spk, occ, valid)

def ratemap(x, y, Xp, Yp, bins=None, min_occ=None, freq=30.0,
    mask_value=np.nan):
    """Construct a 2D spatial rate map for spike and occupancy data.

    The returned matrix contains firing rates or nan for bins with less
    total occupancy than `min_occ`.

    Arguments:
    x, y -- spike position arrays
    Xp, Yp -- occupancy position arrays

    Keyword arguments:
    bins -- int or 2-tuple, number of spatial bins
    min_occ -- minimum sampling occupancy to include a given bin
    freq -- sampling frequency of the occupancy data
    mask_value -- value to set excluded bins

    Returns:
    2D `bins`-shaped array, with x-axis along the first dimension.
    """
    smap = spikemap(x, y, bins=bins)
    omap = occmap(Xp, Yp, bins=bins, min_occ=min_occ, freq=freq)
    valid = np.logical_not(np.isnan(omap))

    H = np.zeros_like(omap) + mask_value
    H[valid] = smap[valid] / omap[valid]

    return H

def spikemap(x, y, bins=None):
    """Spatial spike-count map."""
    bins = bins or DEFAULT_BINS
    return np.histogram2d(x, y, bins=bins, range=NORM_POS_RANGE)[0]

def occmap(x, y, bins=None, min_occ=None, freq=30.0):
    """Spatial occupancy map in seconds.

    Invalid (hyposampled) pixels are set to `np.nan`.
    """
    min_occ = DEFAULT_MIN_OCC if min_occ is None else min_occ
    min_samples = int(min_occ * freq)
    bins = bins or DEFAULT_BINS

    H_occ = np.histogram2d(x, y, bins=bins, range=NORM_POS_RANGE)[0]
    valid = H_occ > min_samples

    H = np.zeros_like(H_occ) + np.nan
    H[valid] = H_occ[valid] / freq

    return H

def dirinfo(d, Dp):
    """Compute the spike directional information rate."""
    smap = dirspikemap(d, bins=INFO_DIR_BINS)
    omap = diroccmap(Dp, bins=INFO_DIR_BINS)[0]
    valid = omap > 0
    return skaggs(smap, omap, valid)

def dirmap(d, Dp, bins=None, freq=30.0):
    """Construct a 1D movement-direction firing-rate map.

    Arguments:
    d -- spike direction array
    Dp -- movement-direction occupancy array

    Keyword arguments:
    bins -- int or 2-tuple, number of directional bins
    freq -- sampling frequency of the occupancy data

    Returns:
    1D `bins`-shaped array of firing rates, array of angles
    """
    smap = dirspikemap(d, bins=bins)
    omap, angles = diroccmap(Dp, bins=bins, freq=freq)
    valid = omap > 0

    H = np.zeros_like(omap)
    H[valid] = smap[valid] / omap[valid]

    return H, angles

def dirspikemap(ds, bins=None):
    """Directional spike-count map."""
    bins = bins or DEFAULT_DIR_BINS
    return np.histogram(ds, bins=bins, range=MDIR_RANGE)[0]

def diroccmap(d, bins=None, freq=30.0):
    """Directional occupancy map in seconds."""
    bins = bins or DEFAULT_DIR_BINS
    H_occ, edges = np.histogram(d, bins=bins, range=MDIR_RANGE)
    H = H_occ / freq
    centers = (edges[:-1] + edges[1:]) / 2
    return H, centers

def speedinfo(s, Sp):
    """Compute the spike speed information rate."""
    kwds = dict(bins=INFO_SPEED_BINS, smin=INFO_SPEED_MIN, smax=INFO_SPEED_MAX)
    smap = speedspikemap(s, **kwds)
    omap = speedoccmap(Sp, min_occ=INFO_SPEED_MIN_OCC, **kwds)[0]
    valid = np.logical_not(np.isnan(omap))
    return skaggs(smap, omap, valid)

def speedmap(s, Sp, bins=None, smin=None, smax=None, min_occ=None, freq=30.0,
    mask_value=np.nan):
    """Construct a speed firing-rate map.

    Arguments:
    s -- spike speed array
    Sp -- occupancy speed array

    Keyword arguments:
    bins -- int or 2-tuple, number of speed bins
    smin/smax -- minimum/maximum velocity of the speed map in cm/s
    min_occ -- minimum sampling occupancy to include a given bin
    freq -- sampling frequency of the occupancy data
    mask_value -- value to set excluded bins

    Returns:
    1D `bins`-shaped array of firing rates, array of bin speeds
    """
    kwds = dict(bins=bins, smin=smin, smax=smax)
    H_spikes = speedspikemap(s, **kwds)
    H_occ, centers = speedoccmap(Sp, min_occ=min_occ, freq=freq, **kwds)
    valid = np.logical_not(np.isnan(H_occ))

    H = np.zeros_like(H_occ) + mask_value
    H[valid] = H_spikes[valid] / H_occ[valid]

    return H, centers

def speedspikemap(s, bins=None, smin=None, smax=None):
    """Speed spike-count map."""
    bins = bins or DEFAULT_SPEED_BINS
    smin = DEFAULT_SPEED_MIN if smin is None else smin
    smax = DEFAULT_SPEED_MAX if smax is None else smax
    edges = np.linspace(smin, smax, bins + 1)
    #return np.histogram(s, bins=bins, range=(smin, smax))[0]
    return np.histogram(s, bins=edges)[0]

def speedoccmap(s, bins=None, smin=None, smax=None, min_occ=None, freq=30.0):
    """Speed occupancy map in seconds.

    Invalid (hyposampled) pixels are set to `np.nan`.
    """
    bins = bins or DEFAULT_SPEED_BINS
    smin = DEFAULT_SPEED_MIN if smin is None else smin
    smax = DEFAULT_SPEED_MAX if smax is None else smax
    min_occ = DEFAULT_SPEED_MIN_OCC if min_occ is None else min_occ
    min_samples = int(min_occ * freq)

    H_occ, edges = np.histogram(s, bins=bins, range=(smin, smax))
    centers = (edges[:-1] + edges[1:]) / 2
    valid = H_occ > min_samples

    H = np.zeros_like(H_occ) + np.nan
    H[valid] = H_occ[valid] / freq

    return H, centers

def phaseinfo(xs, ys, sphase):
    """Compute the mutual information between firing phase and position."""
    xbins = ybins = INFO_PHASE_POS_BINS
    xmin, xmax = INFO_PHASE_POS_XRANGE
    ymin, ymax = INFO_PHASE_POS_YRANGE
    pmin, pmax = PHASE_MIN, PHASE_MAX
    pbins = INFO_PHASE_BINS

    sample = np.c_[xs, ys, sphase]
    xe = np.linspace(xmin, xmax, xbins + 1)
    ye = np.linspace(ymin, ymax, ybins + 1)
    pe = np.linspace(pmin, pmax, pbins + 1)
    edges = [xe, ye, pe]

    H = np.histogramdd(sample, bins=edges)[0].reshape((-1, pbins))
    Px = _norm(H.sum(axis=-1))
    Pp = _norm(H.sum(axis=0))
    Pxp = _norm(H)
    Pmarg = _norm(np.outer(Px, Pp))

    valid = np.logical_and(Pxp != 0.0, Pmarg != 0.0)
    joint = Pxp[valid]
    marginal = Pmarg[valid]

    I = np.sum(joint * np.log2(joint / marginal))

    return I

def phasedist1d(xs, sphase, xmin=0, xmax=1):
    """Construct a 1D joint phase-space probability distribution."""
    xbins = INFO_PHASE_POS_BINS
    pmin, pmax = PHASE_MIN, PHASE_MAX
    pbins = INFO_PHASE_BINS

    xe = np.linspace(xmin, xmax, xbins + 1)
    pe = np.linspace(pmin, pmax, pbins + 1)
    edges = [xe, pe]

    H, _, _ = np.histogram2d(xs, sphase, bins=edges)
    Hx = H.sum()
    if Hx == 0:
        return H
    P = H / Hx
    return P

def phasedist2d(xs, ys, sphase, xmin=0, xmax=CIRCLE_DIAMETER, ymin=0,
    ymax=CIRCLE_DIAMETER):
    """Construct a 2D joint phase-space probability distribution.

    Arguments:
    xs, ys -- (x,y) arrays of spike positions
    sphase -- array of phase at each spike position
    xmin/xmax -- data limits along x dimension
    ymin/ymax -- data limits along y dimension

    Returns:
    3D (xbins,ybins,pbins)-shaped array
    """
    xbins = INFO_PHASE_POS_BINS
    pbins = INFO_PHASE_BINS
    pmin, pmax = PHASE_MIN, PHASE_MAX

    xe = np.linspace(xmin, xmax, xbins + 1)
    ye = np.linspace(ymin, ymax, xbins + 1)
    pe = np.linspace(pmin, pmax, pbins + 1)
    edges = [xe, xe, pe]

    sample = np.c_[xs, ys, sphase]
    H, _ = np.histogramdd(sample, bins=edges)
    Hx = H.sum()
    if Hx == 0:
        return H
    P = H / Hx
    return P

def phasemap(xs, ys, sphase, bins=None, min_spikes=None, mask_value=np.nan):
    """Construct a 2D spatial spike-phase map.

    Arguments:
    xs, ys -- (x,y) arrays of spike positions
    sphase -- array of phase at each spike position

    Keyword arguments:
    bins -- number of bins for each spatial dimension
    min_spikes -- minimum sampling of spikes to include a given bin
    mask_value -- value to set excluded bins

    Returns:
    3D (2,bins,bins)-shaped array, with x-axis along the second dimension.
    """
    min_spikes = DEFAULT_PHASE_MIN_SPIKES if min_spikes is None else min_spikes
    bins = bins or DEFAULT_BINS

    nrange = NORM_POS_RANGE
    x = np.linspace(nrange[0][0], nrange[0][1], bins + 1)
    y = np.linspace(nrange[1][0], nrange[1][1], bins + 1)

    H = np.zeros((2, bins, bins)) + mask_value
    for i, j in product(range(bins), range(bins)):
        ix = np.logical_and(
                np.logical_and(xs >= x[i], xs < x[i+1]),
                np.logical_and(ys >= y[j], ys < y[j+1]))
        if ix.sum() < min_spikes:
            continue
        phase = sphase[ix]
        H[:,i,j] = circstats.mean_vector(phase)

    return H
