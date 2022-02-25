"""
Adaptive estimation spatial maps for firing rate and theta phase.
"""

from scipy import signal
import numpy as np

from pouty import debug
from roto import circstats, arrays
from roto.decorators import lazyprop, datamemoize

from . import kernels
from ..lib.motion import NORM_POS_EXTENT, NORM_POS_MAX


NPIXELS = 64
NBR_FRAC = 0.04
MINRAD = 8.0
MAXRAD = 30.0
MASK_SMOOTHING = 1/15
CM_SCALE = 0.8

@datamemoize
def smooth_mask2d(x, y, bins=NPIXELS, extent=NORM_POS_EXTENT,
    smoothing=MASK_SMOOTHING, scale_max=NORM_POS_MAX):
    """Generate 2D fill mask based on a smoothed histogram."""
    debug('generating arena mask')
    l, r, b, t = extent
    w, h = r - l, t - b

    try:
        nx, ny = bins
    except TypeError:
        nx = ny = int(bins)
    finally:
        n = np.sqrt(nx * ny)
        s = smoothing / (np.max((w, h)) / scale_max)
        k = max(int(s * n), 3)
        k = (k % 2) and k or k + 1

    dx, dy = w / nx, h / ny
    xedges = np.linspace(l - k * dx, r + k * dy, nx + 2 * k + 1)
    yedges = np.linspace(b - k * dy, t + k * dy, ny + 2 * k + 1)

    H = np.histogram2d(x, y, bins=[xedges, yedges])[0]
    H = signal.medfilt2d((H>0).astype('d'), kernel_size=3)

    M = np.zeros_like(H, 'i')
    r, c = H.shape
    for i in range(r):
        j = 0
        while j < c and H[i,j] == 0.0:
            M[i,j] += 1
            j += 1
        j = c - 1
        while j >= 0 and H[i,j] == 0.0:
            M[i,j] += 1
            j -= 1
    for j in range(c):
        i = 0
        while i < r and H[i,j] == 0.0:
            M[i,j] += 1
            i += 1
        i = r - 1
        while i >= 0 and H[i,j] == 0.0:
            M[i,j] += 1
            i -= 1

    Ms = signal.medfilt2d((M<2).astype('d'), kernel_size=k) == 0
    return Ms[k:-k,k:-k]


class AbstractAdaptiveMap(object):

    def __init__(self, mdata, scale='norm', nfrac=NBR_FRAC, alim=(MINRAD, MAXRAD),
        res=NPIXELS, extent=NORM_POS_EXTENT, mask_value=np.nan):
        """Compute spatial maps using adaptive Gaussian kernels.

        Arguments:
        mdata -- a MotionData object for the trajectory being mapped

        Keyword arguments:
        scale -- 'norm'|'cm', set to 'cm' if using map for cm-scaled data
        nfrac -- fraction of data points that constitute a neighborhood
        alim -- adaptive range limits for the kernel radius
        res -- pixel resolution of the output maps (in pixel rows)
        extent -- scalars (left, right, bottom, top), map data extent
        mask_value -- value for setting masked pixels

        Returns a callable that produces spatial maps.
        """
        if hasattr(mdata, 'motion'):
            mdata = mdata.motion  # get motion data if session passed in
        self.mdata = mdata

        assert scale in ('norm', 'cm'), 'scale must be in ("norm", "cm")'
        self.scaled = scale == 'cm'
        if self.scaled:
            if alim == (MINRAD, MAXRAD):
                alim = (CM_SCALE * MINRAD, CM_SCALE * MAXRAD)
            if extent == NORM_POS_EXTENT:
                extent = tuple(map(lambda x: CM_SCALE * x, NORM_POS_EXTENT))
            debug('adaptive ratemap scaled to cm')

        self.nfrac = nfrac
        self.alim = alim
        self.res = res
        self.extent = extent
        self.mask_value = mask_value
        self._cache = {}

    def _get_dataset(self, X, Y):
        pts = np.atleast_2d((X, Y))
        if pts.shape[0] == 2:
            pts = pts.T
        return pts

    def _reshape_grid(self, pts):
        m = self.arena_mask.flatten()
        grid = np.zeros(m.size)
        grid[m] = self.mask_value
        grid[np.logical_not(m)] = pts
        grid = np.reshape(grid, self.pixel_shape)
        return grid

    @lazyprop
    def aspect_ratio(self):
        x = self.extent
        return (x[1] - x[0]) / (x[3] - x[2])

    @lazyprop
    def pixel_shape(self):
        return int(self.aspect_ratio * self.res), self.res

    @lazyprop
    def eval_pixels(self):
        """Compute the pixel grid of kernel evaluation points."""
        _nx, _ny = self.pixel_shape
        _x = np.linspace(self.extent[0], self.extent[1], _nx)
        _y = np.linspace(self.extent[2], self.extent[3], _ny)
        X, Y = np.meshgrid(_x, _y)
        pixels = np.c_[X.T.flatten(), Y.T.flatten()]
        test = np.logical_not(self.arena_mask.flatten())
        return pixels[test]

    @lazyprop
    def arena_mask(self):
        """Generate map mask based on whole trajectory."""
        if self.scaled:
            xdata, ydata = self.mdata.x_cm, self.mdata.y_cm
            scale_max = CM_SCALE * NORM_POS_MAX
        else:
            xdata, ydata = self.mdata.x, self.mdata.y
            scale_max = NORM_POS_MAX
        return smooth_mask2d(xdata, ydata, bins=self.pixel_shape,
                extent=self.extent, scale_max=scale_max)

    def knbrs(self, N):
        try:
            N = N.shape[0]
        except AttributeError:
            pass
        return max(1, int(N * self.nfrac))

    def __call__(self, *data):
        """Subclasses must override this to evaluate their kernels."""
        raise NotImplementedError


class _SpikeCountMap(AbstractAdaptiveMap):

    def __call__(self, xs, ys):
        """Compute the rate map with supplied spike and position data."""
        _hash = arrays.datahash(xs, ys)
        if _hash in self._cache:
            return self._cache[_hash]

        debug('running kernel estimation for spikes')
        spkdata = self._get_dataset(xs, ys)
        k = kernels.AdaptiveGaussianKernel(spkdata,
                k_neighbors=self.knbrs(spkdata))
        P_spk = k(self.eval_pixels, minrad=self.alim[0], maxrad=self.alim[1])
        G_spk = self._reshape_grid(P_spk) * spkdata.shape[0]  # scale spike estimate
        self._cache[_hash] = G_spk
        return G_spk


class _OccupancyMap(AbstractAdaptiveMap):

    def __call__(self, xp, yp, Fs=None):
        """Compute the rate map with supplied spike and position data."""
        _hash = arrays.datahash(xp, yp)
        if _hash in self._cache:
            return self._cache[_hash]
        if Fs is None:
            Fs = self.mdata.Fs

        debug('running kernel estimation for occupancy')
        posdata = self._get_dataset(xp, yp)
        duration = posdata.shape[0] / Fs
        k = kernels.AdaptiveGaussianKernel(posdata,
                k_neighbors=self.knbrs(posdata))
        P_occ = k(self.eval_pixels, minrad=self.alim[0], maxrad=self.alim[1])
        G_occ = self._reshape_grid(P_occ) * duration  # scale occupancy estimate
        self._cache[_hash] = G_occ
        return G_occ


class AdaptiveRatemap(object):

    """
    Manage spike-count and occupancy estimates to compute firing-rate maps.
    """

    def __init__(self, *args, **kwargs):
        self._spkmap = _SpikeCountMap(*args, **kwargs)
        self._occmap = _OccupancyMap(*args, **kwargs)
        self.mask_value = kwargs.get('mask_value', np.nan)

    def __call__(self, xs, ys, xp, yp, Fs=None):
        if xs.size == 0:
            G_spk = np.zeros(self._spkmap.pixel_shape)
        else:
            G_spk = self._spkmap(xs, ys)
        G_occ = self._occmap(xp, yp, Fs=Fs)
        G_rate = np.zeros_like(G_spk) + self.mask_value
        valid = np.logical_and(np.isfinite(G_spk), np.isfinite(G_occ))
        G_rate[valid] = G_spk[valid] / G_occ[valid]
        return G_rate


def phase_vector(weights, values):
    """Multi-output kernel function to compute mean phase vectors."""
    return circstats.mean_resultant_vector(values, weights=weights)

class AdaptivePhasemap(AbstractAdaptiveMap):

    """
    Manage occupancy and phase estimates for adaptive phase maps.
    """

    def __call__(self, xs, ys, phase):
        """Compute the phase mean and spread estimates on bursting data."""
        _hash = arrays.datahash(xs, ys, phase)
        if _hash in self._cache:
            return self._cache[_hash]
        if xs.size == 0:
            self._cache[_hash] = np.zeros(self.pixel_shape) + self.mask_value
            return self._cache[_hash]

        debug('running kernel estimation of spike phase')
        posdata = self._get_dataset(xs, ys)
        k = kernels.AdaptiveGaussianKernel(posdata,
                k_neighbors=self.knbrs(posdata),
                values=phase)
        L_phase = k(self.eval_pixels, minrad=self.alim[0], maxrad=self.alim[1],
                kernel_func=phase_vector, n_outputs=2)
        G_phase = np.empty((2,) + self.pixel_shape)
        G_phase[0] = self._reshape_grid(L_phase[0])
        G_phase[1] = self._reshape_grid(L_phase[1])
        self._cache[_hash] = G_phase
        return G_phase


def weighted_avg(weights, values):
    """Compute a weighted average across neighbor values."""
    totw = np.sum(weights, axis=-1)
    return (weights * values).sum(axis=-1) / totw

class AdaptiveAveragerMap(AbstractAdaptiveMap):

    """
    Compute a local weighted average of values across nearest neighbors.
    """

    def __call__(self, xp, yp, values):
        _hash = arrays.datahash(xp, yp, values)
        if _hash in self._cache:
            return self._cache[_hash]

        debug('running weighted averager on values')
        posdata = self._get_dataset(xp, yp)
        k = kernels.AdaptiveGaussianKernel(posdata,
                k_neighbors=self.knbrs(posdata),
                values=values)
        L_avg = k(self.eval_pixels, minrad=self.alim[0], maxrad=self.alim[1],
                kernel_func=weighted_avg)
        G_avg = self._reshape_grid(L_avg)
        self._cache[_hash] = G_avg
        return G_avg
