"""
The kernels for kernel minisum approach. see the original paper for the
description of the implemented ExponentiatedCosine and SquareAngle kernels.

If a new kernel needed, it should inherit Kernel class, and provide the
following functional signatures:
    score(w, mask=None) -> [scores]
    gradient(w, old_w=None, old_dw=None, normal=None) -> w, dw

"""

import numpy as np

class Kernel():

    def __init__(self, x, g):

        self.x = x
        self.g = g
        self.e = self.x / np.linalg.norm(self.x, axis=1, keepdims=True)


    def _preprocess_args(w, mask, N):

        return (w[None, :] if len(w.shape) == 1 else w,
                np.ones(N, dtype=bool) if mask is None else mask)


    def _get_alpha(w, dw, old_w, old_dw):

        if old_w is None:
            return 0.1

        ddw = dw - old_dw
        return abs((w - old_w) @ ddw / np.linalg.norm(ddw) ** 2)


class ExponentiatedCosine(Kernel):

    def score(self, w, mask=None):

        w, mask = Kernel._preprocess_args(w, mask, len(self.g))

        return (self.g[mask] *
                np.exp(-w @ self.e[mask].T + 1) - 1).sum(axis=1)

    def gradient(self, w, old_w=None, old_dw=None, normal=None):

        dw = np.sum(-(self.g * np.exp(-w @ self.e.T))[:, None] * self.e, axis=0)
        dw -= (dw @ w) * w

        # restrict the search if normal is provided
        if normal is not None:
            dw -= normal * (dw @ normal)

        wnew = w - Kernel._get_alpha(w, dw, old_w, old_dw) * dw

        # fix possible rounding errors in restricted case
        if normal is not None:
            wnew -= normal * (wnew @ normal)

        return wnew / np.linalg.norm(wnew), dw


class SquareAngle(Kernel):


    def score(self, w, mask=None):

        w, mask = Kernel._preprocess_args(w, mask, len(self.g))

        return (self.g[mask] * 0.5 *
                np.arccos(w @ self.e[mask].T) ** 2).sum(axis=1)


    def gradient(self, w, old_w=None, old_dw=None, normal=None):

        wxt = w @ self.e.T

        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            C = np.arccos(wxt) / np.sqrt(1 - wxt ** 2)

        C[wxt >  0.9990] = 1.0
        C[wxt < -0.9848] = 17.0866


        dw = np.sum((-self.g * C)[:, None] * self.e, axis=0)
        dw -= (dw @ w) * w

        # restrict the search if normal is provided
        if normal is not None:
            dw -= normal * (dw @ normal)

        wnew = w - Kernel._get_alpha(w, dw, old_w, old_dw) * dw

        # fix possible rounding errors in restricted case
        if normal is not None:
            wnew -= normal * (wnew @ normal)

        return wnew / np.linalg.norm(wnew), dw
