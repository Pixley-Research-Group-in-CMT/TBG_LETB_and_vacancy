#!/usr/bin/env python
"""Sanity checks for an assembled vacancy model (and its pristine host).

    python check_model.py models/theta_1.05_AA

Reports
  * Hermiticity of every model file in the folder;
  * the pristine host's charge-neutrality point E_CNP (Dirac point at the moire K point) and the
    width of the four central bands sampled at Gamma, K, M;
  * for each spin: where the spectral weight of the middle sigma mode (the V_sigma state of the
    paper) sits relative to E_CNP, and the spin splitting of that level.
Uses sparse shift-invert diagonalisation, so it also runs for 1.05 deg (about a minute).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tbmodel import TBModel  # noqa: E402

# high-symmetry points in reciprocal-lattice fractions for the packaged cells
# (a1 = (L,0,0), a2 = (-L/2, L*sqrt(3)/2, 0)  ->  b1, b2 at 60 degrees)
KPTS = {"Gamma": (0.0, 0.0), "K": (1 / 3, 1 / 3), "M": (0.5, 0.0)}


def eigs_near(H, e0, k):
    w, v = sla.eigsh(H, k=k, sigma=e0, which="LM")
    o = np.argsort(w)
    return w[o], v[:, o]


def pristine_report(host):
    # Dirac point: four nearly degenerate levels at K; start from the LETB estimate -3*t2 ~ -0.69 eV
    w, _ = eigs_near(host.bloch_hamiltonian(KPTS["K"]), -0.69, 24)
    quads = [(w[i + 3] - w[i], abs(w[i : i + 4].mean() + 0.69), i) for i in range(len(w) - 3)]
    tight = [q for q in quads if q[0] < 2e-3] or sorted(quads)[:1]
    i = min(tight, key=lambda q: q[1])[2]          # 4-fold Dirac point closest to the estimate
    ecnp = float(w[i : i + 4].mean())
    central = []
    for name, k in KPTS.items():
        wk, _ = eigs_near(host.bloch_hamiltonian(k), ecnp, 8)
        central.append(wk[np.argsort(np.abs(wk - ecnp))[:4]])
    central = np.concatenate(central)
    return ecnp, float(central.max() - central.min())


def sigma_report(m, ecnp, nev=60):
    sg = np.where(m.extra["orbital_kind"] == "sigma")[0]
    H0 = m.bloch_hamiltonian((0, 0))
    hs = H0[sg][:, sg].toarray()
    lev, vec = np.linalg.eigh(hs)
    u = vec[:, 1]
    E, W = [], []
    for k in KPTS.values():
        w, v = eigs_near(m.bloch_hamiltonian(k), lev[1], nev)
        E.append(w)
        W.append(np.abs(u.conj() @ v[sg, :]) ** 2)
    E, W = np.concatenate(E), np.concatenate(W)
    peak = E[np.argmax(W)]
    captured = W.sum() / len(KPTS)
    return lev - ecnp, peak - ecnp, captured


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    a = ap.parse_args()
    folder = Path(a.folder)
    meta = json.loads((folder / "metadata.json").read_text())
    host = TBModel.load(meta["host"])
    print(f"{folder}: {meta['config']} vacancy, removed host site {meta['removed_host_site']}, "
          f"pi block = {meta['pi_block']}, position mismatch max {meta['position_mismatch_A']['max']:.3f} A")
    ecnp, width = pristine_report(host)
    print(f"pristine host: {host.norb} orbitals, E_CNP = {ecnp:+.4f} eV, central-band spread (Gamma,K,M) = {1000*width:.1f} meV")
    peaks = {}
    for spin in ("up", "dn"):
        p = folder / f"model_{spin}.npz"
        if not p.exists():
            continue
        m = TBModel.load(p)
        herm = m.hermiticity_error()
        line = f"  {spin}: {m.norb} orbitals, hermiticity error {herm:.1e}"
        if (m.extra["orbital_kind"] == "sigma").any():
            lev, peak, cap = sigma_report(m, ecnp)
            peaks[spin] = peak
            line += (f"; sigma-block levels - E_CNP = {np.round(lev, 3).tolist()} eV; "
                     f"V_sigma weight peak at {peak:+.3f} eV (weight captured {cap:.2f})")
        print(line)
    if len(peaks) == 2:
        print(f"  V_sigma spin splitting (dn - up) = {peaks['dn'] - peaks['up']:.3f} eV")
    print("  neutral filling: total electrons (both spins) = number of orbitals in one spin model "
          f"= {m.norb}; the Fermi level of a periodic vacancy model must be computed for that filling.")


if __name__ == "__main__":
    main()
