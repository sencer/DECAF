import numpy as np


class Quad:


    def __init__(self, points, weights):

        self.points = points
        self.weights = weights



def compose(*args):

    points = []
    weights = []

    for args in args:

        points.append(args.points)
        weights.append(args.weights)


    return Quad(np.column_stack(points),
                np.column_stack(weights))


def spherical_shells(radial, angular):

    try:
        assert len(angular) == len(radial.points)
    except:
        angular = [angular] * len(radial.points)

    points = []
    weights = []

    for r, w, a in zip(radial.points, radial.weights, angular):

        points.append(a.points * r)
        weights.append(a.weights * w)

    return Quad(np.row_stack(points),
                np.hstack(weights))
