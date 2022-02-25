"""
Functions for computing spatial relations and weights.
"""

import pandas as pd
import numpy as np
import scipy.stats as st

from pouty import debug
from roto.decorators import lazyprop
from roto.radians import cdiff

from ..ana.phaser_model import DEFAULT_F_THETA
from ..lib.motion import CIRCLE_AND_BOX_SIZE as SIZE

ARENA_EXTENT = [0, SIZE, 0, SIZE]
XMIN, XMAX, YMIN, YMAX = ARENA_EXTENT
GNORM = 1 / (np.sqrt(2*np.pi))

THETA_WINDOW = 1 / DEFAULT_F_THETA


class BayesPhaseDecoder(object):

    def __init__(self, phasemaps, xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX):
        self.P = phasemaps
        self.P[np.isnan(self.P)] = 0.0
        self.N = phasemaps.shape[0]
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.ngrid = phasemaps.shape[1]
        self.argmax = None

    def _validate_activation(self, H):
        H = np.atleast_1d(np.squeeze(H))
        assert H.ndim == 1 and H.size == self.N, 'size or dimension mismatch'
        assert np.all(np.isfinite(H)), 'invalid activation values'
        return H

    def decode(self, spike_phase, window=THETA_WINDOW, tau=1, continuity=0.0):
        """Bayesian posterior for decoding windowed spike-phase averages."""
        H = self._validate_activation(spike_phase)
        L = np.prod(np.exp(np.cos(cdiff(H.reshape(-1,1,1), self.P))), axis=0)
        P = L * np.exp(-window * tau)
        P /= np.trapz(np.trapz(P, x=self._y_bins), x=self._x_bins)

        if continuity > 0 and self.argmax is not None:
            dist2 = (self._eval_grid[0]-self.argmax[0])**2 + \
                    (self._eval_grid[1]-self.argmax[1])**2
            prior = (GNORM/continuity) * np.exp(-dist2/(2*continuity)**2)
            P *= prior
            P /= np.trapz(np.trapz(P, x=self._y_bins), x=self._x_bins)

        # Save spatial argmax for continuity constraint
        self.argmax = self._argmax_ij(P)

        return P

    def _argmax_ij(self, P):
        """Find the spatial coordinates (ij-index) for the maximum of a map."""
        YY, XX = self._eval_grid
        i = P.ravel().argmax()
        return YY.ravel()[i], XX.ravel()[i]

    @lazyprop
    def _eval_grid(self):
        """Mesh grid for Poisson sampling and evaluations."""
        return np.meshgrid(self._x_bins, self._y_bins, indexing='ij')

    @lazyprop
    def _x_bins(self):
        aspect = (self.xmax - self.xmin) / (self.ymax - self.ymin)
        return np.linspace(self.xmin, self.xmax, int(aspect*self.ngrid))

    @lazyprop
    def _y_bins(self):
        return np.linspace(self.ymin, self.ymax, self.ngrid)
