import numpy as np

from quadpy.sphere import Lebedev
from quadpy.e1r import GaussLaguerre

from .quad import spherical_shells


def GaussLaguerre_Lebedev(nradial, shells, cutoff, scale=0.9, alpha=2):
    """ returns the quadrature described in the original paper
    """

    assert nradial == len(shells)

    radial = GaussLaguerre(nradial, alpha)

    norm = cutoff * scale / radial.points.max()

    radial.weights *= np.exp(radial.points) * norm**3
    radial.points *= norm

    quad = spherical_shells(radial, [Lebedev(shell) for shell in shells])
    quad.weights *= 4 * np.pi

    return quad


def minmaxscaler(arr):

    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo)


def blurring_gaussians(alpha, scale=1, shift=0.5):

    def sigma(pos, masses, radii, cutoff):
        if scale is None:
            r_ = radii
        else:
            r_ = minmaxscaler(radii) * scale + shift

        r = np.linalg.norm(pos, axis=1)
        return r_ * (1 + alpha * r)

    return sigma


def weighed_kernel(kernel=None, scale=1, shift=0.5):

    if kernel is None:
        kernel = tent

    def intensity(pos, masses, radii, cutoff):

        # normalize masses to [shift, shift+scale] range
        m = minmaxscaler(masses) * scale + shift

        r = np.linalg.norm(pos, axis=1)

        return m * kernel(r/cutoff)

    return intensity


def tent(rnorm, t=3):
    return (1 - rnorm)**t

def bell(rnorm, a=4, b=3):
    return (a * tent(rnorm, b) - b * tent(rnorm, b)) / (a-b)
