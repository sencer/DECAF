import numpy as np

from ase.neighborlist import NeighborList
from ase.data import covalent_radii as rcov
from .rotation import CanonicalRotation


class Fingerprint(np.ndarray):

    def __new__(cls, arr, generator):

        obj = np.asarray(arr).view(cls)
        obj.generator = generator

        return obj


    def __sub__(self, other):

        if callable(self.generator.w_integral):
            w = self.generator.w_integral(self.generator.sensors,
                                        self.generator.cutoff)
        else:
            w = 1

        return np.linalg.norm((np.asarray(self) - np.asarray(other)) * w)


    def plot(self, ax=None, mode=0, **kwargs):

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        if mode == 0:
            ax.plot(self, **kwargs)
            ax.set_xlabel("Sensor #")
            ax.set_ylabel("Intensity")
            ax.set_yticks(())
        if mode == 1:
            rs = np.linalg.norm(self.generator.sensors, axis=1)
            ax.scatter(rs, self, **kwargs)
            ax.set_xlabel("r, $\mathrm{\AA}$")
            ax.set_ylabel("Intensity")
            ax.set_yticks(())

        return ax


    def plotshell(self, r, dr, ax=None, **kwargs):

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        rs = np.linalg.norm(self.generator.sensors, axis=1)
        mask = (rs >= r - dr) & (rs <= r + dr)

        if not np.any(mask):
            raise Exception("No sensors in given r ± dr range.")

        r = rs[mask]
        p = self.generator.sensors[mask]

        xy = np.linalg.norm(p[:, :2], axis=1)
        elevation = np.arctan2(xy, p[:, 2])
        azimuth = np.arctan2(p[:, 1], p[:, 0])

        if 'c' in kwargs:
            c = kwargs.pop('c')[mask]
        else:
            c = self[mask]

        im = ax.scatter(azimuth, elevation, c=c, **kwargs)

        ax.set_xlabel('Azimuth')
        ax.set_xlim(-np.pi - 0.3, np.pi + 0.3)
        ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2, np.pi))
        ax.set_xticklabels(('-π', '-π/2', '0', 'π/2', 'π'))

        ax.set_ylabel('Elevation')
        ax.set_ylim(-0.3, np.pi + 0.3)
        ax.set_yticks((0, np.pi/2, np.pi))
        ax.set_yticklabels(('0', 'π/2', 'π'))

        return ax, im


class FPGenerator():

    def __init__(self, quadrature, cutoff, atoms=None, I=5E-2, sigma=2,
                 g=None, w=1, debug=False):

        self.g = g
        self.I = I
        self.sigma = sigma
        self.cutoff = cutoff
        self.debug = debug

        self.quad = quadrature
        self.weights = self.quad.weights
        self.sensors = self.quad.points

        self.w_integral = w

        self.atoms = atoms
        if atoms is not None:
            self.nlist = NeighborList([cutoff * 0.5] * len(atoms), skin=0,
                                      self_interaction=False, bothways=True)

            self.nlist.update(atoms)

    def materialize(self, atoms):
        return FPGenerator(self.quad, self.cutoff, atoms=atoms, I=self.I,
                           sigma=self.sigma, g=self.g, w=self.w_integral)

    def __getitem__(self, iatom):
        # TODO let it work with slices

        if self.atoms == None or len(self.atoms) <= iatom:
            raise Exception("Check if atoms object exist or has %d atoms." % 
                            (iatom + 1))

        # get all the atoms in the environment, offsets keep the periodicity
        # information
        neighbors, offsets = self.nlist.get_neighbors(iatom)

        # positions (relative to the central atom), covalent radii and
        # atomic masses will be used to calculate the "importance" of atoms
        # in kernel minisum; intensity and spread of the gaussians associated
        # with each atom

        # calculate atomic positions relative to the atom of interest
        pos = (self.atoms.positions[neighbors] +
               offsets @ self.atoms.cell) - self.atoms[iatom].position

        # get the covalent radii
        radii = rcov[self.atoms.numbers[neighbors]]

        # and masses
        masses = self.atoms.get_masses()[neighbors]

        # calculate g
        if callable(self.g):
            g = self.g(pos, masses, radii, self.cutoff)
        else:
            g = masses * np.exp(-np.linalg.norm(pos, axis=1))

        # evaluate sigma for each atom; if not callable it is 
        # sigma * covalent radius of the atom
        if callable(self.sigma):
            sigma = self.sigma(pos, masses, radii, self.cutoff)
        else:
            sigma = self.sigma * radii

        # evaluate I for each atom; if not callable it is 
        # I * mass of the atom
        if callable(self.I):
            I = self.I(pos, masses, radii, self.cutoff)
        else:
            I = self.I * masses

        #   set stage for minisum calculation to find the canonical coordinates 
        rotator = CanonicalRotation(pos, g, debug=self.debug)

        #   rotate sensors
        sensors = self.sensors @ rotator.Rinv

        if self.debug:
            from ase.io import write
            from ase.atoms import Atoms

            write("%s_%d.xyz" % (self.atoms.get_chemical_formula(), iatom),
                  Atoms(self.atoms.numbers[neighbors],
                        positions=pos @ rotator.R) +
                  Atoms([2]*len(sensors), positions=self.sensors))


        # calculate (R_atom - r_sensor)^2
        r = np.linalg.norm(sensors[:, None] - pos, axis=2)

        # return a vector vec length len(sensors) where vec[j] is:
        # \Sum_{i} w_j * (I * exp(-(R_i - r_j)^2/sigma))
        # i runs over atoms, R is the position of an atom, r is the positon of
        # a sensor.
        # I and sigma are functions of atomic mass and covalent radius
        return Fingerprint(((I * np.exp(-0.5 * (r/sigma)**2) / sigma) *
                             self.weights[:, None]).sum(axis=1), self)
