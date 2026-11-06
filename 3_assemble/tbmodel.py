"""Minimal tools for the tight-binding files in this package (NumPy + SciPy only).

File format (.npz), shared by the pristine hosts and the assembled vacancy models:
    i, j        (m,)   int      orbital indices of a directed hopping t_ij(R) = <0,i|H|R,j>
    R           (m,3)  int      translation in units of the rows of cell_A (R[:,2] == 0)
    t_eV        (m,)   complex  hopping amplitude
    onsite_eV   (n,)   float    on-site energies
    positions_A (n,3)  float    orbital positions in the home cell (Angstrom)
    cell_A      (3,3)  float    lattice vectors as rows (Angstrom)
Both (i,j,R) and (j,i,-R) are stored, so

    H_ij(k) = onsite_i delta_ij + sum_R t_ij(R) exp(2 pi i k.R)       (k in reciprocal-lattice fractions)

Never add the Hermitian conjugate a second time.
"""
import json

import numpy as np
import scipy.sparse as sp


class TBModel:
    def __init__(self, i, j, R, t, onsite, positions, cell, info=None, **extra):
        self.i = np.asarray(i, dtype=np.int64)
        self.j = np.asarray(j, dtype=np.int64)
        self.R = np.asarray(R, dtype=np.int64)
        self.t = np.asarray(t, dtype=complex)
        self.onsite = np.asarray(onsite, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.cell = np.asarray(cell, dtype=float)
        self.info = dict(info or {})
        self.extra = extra

    @property
    def norb(self):
        return len(self.onsite)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["info"])) if "info" in d.files else {}
        core = {"i", "j", "R", "t_eV", "onsite_eV", "positions_A", "cell_A", "info"}
        extra = {k: d[k] for k in d.files if k not in core}
        return cls(d["i"], d["j"], d["R"], d["t_eV"], d["onsite_eV"], d["positions_A"], d["cell_A"], info, **extra)

    def save(self, path):
        np.savez_compressed(path, i=self.i, j=self.j, R=self.R, t_eV=self.t, onsite_eV=self.onsite,
                            positions_A=self.positions, cell_A=self.cell, info=json.dumps(self.info), **self.extra)

    def to_csv(self, prefix):
        """Write <prefix>_hoppings.csv and <prefix>_orbitals.csv (language-independent copies)."""
        with open(f"{prefix}_hoppings.csv", "w") as f:
            f.write("i,j,R1,R2,R3,t_real_eV,t_imag_eV\n")
            for a, b, r, t in zip(self.i, self.j, self.R, self.t):
                f.write(f"{a},{b},{r[0]},{r[1]},{r[2]},{t.real:.10f},{t.imag:.10f}\n")
        kinds = self.extra.get("orbital_kind", np.array(["pz"] * self.norb))
        with open(f"{prefix}_orbitals.csv", "w") as f:
            f.write("orbital,kind,x_A,y_A,z_A,onsite_eV\n")
            for a, (p, e) in enumerate(zip(self.positions, self.onsite)):
                f.write(f"{a},{kinds[a]},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{e:.10f}\n")

    # ------------------------------------------------------------------ checks
    def hermiticity_error(self):
        key = {(a, b, tuple(r)): t for a, b, r, t in zip(self.i, self.j, self.R, self.t)}
        err = 0.0
        for (a, b, r), t in key.items():
            back = key.get((b, a, (-r[0], -r[1], -r[2])))
            if back is None:
                return np.inf
            err = max(err, abs(t - np.conj(back)))
        return err

    # ------------------------------------------------------------------ matrices
    def bloch_hamiltonian(self, k=(0.0, 0.0), mu_eV=0.0, atomic_gauge=False):
        """Sparse H(k). k = fractional coordinates (k1, k2) of the reciprocal lattice.
        Cell gauge by default; atomic_gauge=True includes orbital positions in the phase."""
        k = np.array([k[0], k[1], 0.0])
        phase = self.R @ k
        if atomic_gauge:
            frac = self.positions @ np.linalg.inv(self.cell)
            phase = phase + (frac[self.j] - frac[self.i]) @ k
        H = sp.coo_matrix((self.t * np.exp(2j * np.pi * phase), (self.i, self.j)), shape=(self.norb,) * 2).tocsr()
        return H + sp.diags(self.onsite - mu_eV)

    def supercell(self, n1, n2):
        """Return the n1 x n2 supercell as a new TBModel (cell rows n1*a1, n2*a2).
        Orbital index = (c1*n2 + c2)*norb + orbital, so copy (0,0) keeps the original indices.
        Use it to build a large *pristine* host and then insert one vacancy into it."""
        n = self.norb
        I, J, RR, T = [], [], [], []
        for c1 in range(n1):
            for c2 in range(n2):
                d1, d2 = c1 + self.R[:, 0], c2 + self.R[:, 1]
                I.append((c1 * n2 + c2) * n + self.i)
                J.append(((d1 % n1) * n2 + d2 % n2) * n + self.j)
                RR.append(np.stack([d1 // n1, d2 // n2, self.R[:, 2]], axis=1))
                T.append(self.t)
        shifts = np.array([c1 * self.cell[0] + c2 * self.cell[1] for c1 in range(n1) for c2 in range(n2)])
        pos = (shifts[:, None, :] + self.positions[None]).reshape(-1, 3)
        cell = self.cell.copy()
        cell[0] *= n1
        cell[1] *= n2
        extra = {k: np.tile(v, n1 * n2) if (np.ndim(v) == 1 and len(v) == n) else v for k, v in self.extra.items()}
        info = dict(self.info, supercell=[n1, n2])
        return TBModel(np.concatenate(I), np.concatenate(J), np.concatenate(RR), np.concatenate(T),
                       np.tile(self.onsite, n1 * n2), pos, cell, info, **extra)

    def realspace_supercell(self, n1, n2, periodic=True, mu_eV=0.0):
        """Real-space Hamiltonian of an n1 x n2 supercell (orbital index = cell*norb + orbital).
        NOTE: a vacancy model repeated this way contains n1*n2 vacancies."""
        n = self.norb
        rows, cols, vals = [], [], []
        for c1 in range(n1):
            for c2 in range(n2):
                d1, d2 = c1 + self.R[:, 0], c2 + self.R[:, 1]
                if periodic:
                    d1, d2, keep = d1 % n1, d2 % n2, np.ones(len(d1), bool)
                else:
                    keep = (d1 >= 0) & (d1 < n1) & (d2 >= 0) & (d2 < n2)
                rows.append((c1 * n2 + c2) * n + self.i[keep])
                cols.append((d1[keep] * n2 + d2[keep]) * n + self.j[keep])
                vals.append(self.t[keep])
        N = n * n1 * n2
        H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
        return H + sp.diags(np.tile(self.onsite - mu_eV, n1 * n2))
