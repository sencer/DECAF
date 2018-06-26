"""
kernel minisum
"""

import numpy as np

from math import sin, cos, sqrt
from quadpy.sphere import Lebedev

from .kernel import ExponentiatedCosine as EC


class CanonicalRotation():


    def __init__(self, x, g, kernel=EC, maxiter=100, convthr=1E-14,
                 n3d=31, n2d=30, debug=False):

        self.x = x
        self.g = g
        self.kernel = kernel(x, g)

        self.n2d = n2d
        self.n3d = n3d

        self.maxiter = maxiter
        self.convthr = convthr
        self.debug = debug

        self.calculate_basis()


    def descent(self, w, normal=None):

        if self.debug:
            print("  i   change   len")
            print("%3d  ******** %5.2f" % (0, self.score(w)[0]))

        converged = False
        old_w = old_dw = None

        for i in range(self.maxiter):

            old_w, (w, old_dw) = w, self.kernel.gradient(w, old_w, old_dw,
                                                         normal)

            diff = np.linalg.norm(w - old_w)

            if self.debug:
                print("%3d %9.2E %5.2f" % (i+1, diff, self.score(w)[0]))

            if diff < self.convthr:
                converged = True
                break

        if converged:
            return w, self.kernel.score(w)[0]
        else:
            return None, 1E15


    def run_minisum(self, normal=None):

        if normal is None:
            ws = Lebedev(self.n3d).points
        else:
            if normal[0] ** 2 + normal[1] ** 2 > 1E-15:
                v = np.array((-normal[1], normal[0], 0))
            else:
                v = np.array((normal[2], 0, -normal[0]))

            v /= np.linalg.norm(v)
            u = np.cross(normal, v)

            ws = [v * sin(theta) + u * cos(theta) for theta in
                      np.linspace(-np.pi, np.pi, self.n2d)]

        vec = None
        best = 1E15
        for w in ws:
            tmp, current = self.descent(w, normal=normal)

            if current < best:
                vec = tmp
                best = current

        if vec is None:
            raise Exception

        return vec


    def calculate_basis(self):

        u = self.run_minisum()
        v = self.run_minisum(normal=u)
        w = np.cross(u, v)


        front = (w @ self.x.T) > 0
        probe = (u+v) * sqrt(0.5)

        score1 = self.kernel.score(probe, mask=front)
        score2 = self.kernel.score(probe, mask=~front)

        self.basis = u, v, w if score1 <= score2 else -w

        return self.basis


    def rotated(self):

        return self.x @ self.R


    @property
    def R(self):
        return self.Rinv.T


    @property
    def Rinv(self):
        return np.vstack(self.basis)
