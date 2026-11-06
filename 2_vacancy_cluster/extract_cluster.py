#!/usr/bin/env python
"""Piece 2: build the vacancy cluster (local tight-binding patch) from the DFT Wannier model.

    python extract_cluster.py AA          # also AB, ABp; reads wannier_inputs/<config>/
    python extract_cluster.py --all

Input  (wannier_inputs/<config>/): Wannier90 output of a 6x6 untwisted bilayer supercell with one
       bottom-layer vacancy (143 C): wannier90.{up,dn}_hr.dat(.gz), *_centres.xyz, *.win (the relaxed
       geometry is read from the .win; POSCAR is a copy for viewing), and POSCAR_novac (ideal lattice
       with the missing atom restored; only used to locate the vacancy site).
Output (clusters/<config>/): cluster.npz plus human-readable CSV/JSON copies.

What is done, in order (README section 2 explains the physics):
 1. Read H(R) for both spins (146 Wannier orbitals: 143 C pz + 3 sp2 on the vacancy neighbours).
 2. On each vacancy-neighbour carbon, rotate the on-site (pz, sp2) pair so that their mutual
    hopping vanishes; the higher level is called pz, the lower one sigma (historical convention).
 3. Gauge: fix the sign of each rotated pz so its nearest-neighbour hoppings have the same sign as
    in the bulk (and as in LETB); make the sp2 coefficient of each sigma orbital real-positive.
 4. Select the patch: bottom-layer carbons within 6 A and top-layer carbons within 4 A (in-plane)
    of the missing site (58 pz) + the 3 sigma orbitals = 61 orbitals.
 5. Write every matrix element between patch orbitals, using the Wannier-centre cell of each
    orbital to find the translation R that realises the *local* bond vector inside the patch.
    The result is a finite 61x61 Hermitian matrix in a local frame centred on the missing site.
 6. Energy reference: shift each layer so that the average pz on-site energy of that layer, far
    (>= 5 A in-plane) from the vacancy, is zero, i.e. equal to the LETB on-site energy. The sigma
    orbitals (bottom layer) get the bottom-layer shift.
"""
import argparse
import gzip
import json
from io import StringIO
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIGS = ("AA", "AB", "ABp")
SPINS = ("up", "dn")
R_BOTTOM, R_TOP, R_FAR = 6.0, 4.0, 5.0


# ----------------------------------------------------------------------------- readers
def _open(path):
    path = Path(path)
    if path.exists():
        return path.open("rt")
    gz = path.with_name(path.name + ".gz")
    return gzip.open(gz, "rt")


def read_poscar(path):
    lines = Path(path).read_text().splitlines()
    scale = float(lines[1])
    cell = np.array([[float(x) for x in lines[k].split()[:3]] for k in (2, 3, 4)]) * scale
    n = sum(map(int, lines[6].split()))
    start = 9 if lines[7].strip().lower().startswith("s") else 8
    mode = lines[start - 1].strip().lower()
    xyz = np.array([[float(x) for x in l.split()[:3]] for l in lines[start : start + n]])
    if mode.startswith("d"):
        xyz = xyz @ cell
    else:
        xyz = xyz * scale
    return cell, xyz


def read_win_structure(path):
    """Cell and Cartesian atom positions from the Wannier90 .win (the geometry Wannier90 actually used)."""
    lines = Path(path).read_text().splitlines()
    low = [l.strip().lower() for l in lines]
    c0 = low.index("begin unit_cell_cart")
    cell = np.array([[float(x) for x in lines[c0 + 1 + k].split()[:3]] for k in range(3)])
    if "begin atoms_cart" in low:
        a0, a1 = low.index("begin atoms_cart"), low.index("end atoms_cart")
        rows = [l.split() for l in lines[a0 + 1 : a1] if l.split()]
        xyz = np.array([[float(x) for x in r[1:4]] for r in rows if r[0][0].isalpha()])
    else:
        a0, a1 = low.index("begin atoms_frac"), low.index("end atoms_frac")
        xyz = np.array([[float(x) for x in l.split()[1:4]] for l in lines[a0 + 1 : a1] if l.split()]) @ cell
    return cell, xyz


def read_hr(path):
    with _open(path) as f:
        lines = f.read().splitlines()
    n, nr = int(lines[1]), int(lines[2])
    nd = (nr + 14) // 15
    deg = np.array([int(x) for s in lines[3 : 3 + nd] for x in s.split()])
    tab = np.loadtxt(lines[3 + nd :]).reshape(nr, n * n, 7)
    R = tab[:, 0, :2].astype(int)
    H = np.zeros((nr, n, n), complex)
    for k, b in enumerate(tab):
        H[k, b[:, 3].astype(int) - 1, b[:, 4].astype(int) - 1] = (b[:, 5] + 1j * b[:, 6]) / deg[k]
    return R, H


def read_centres(path, nwann):
    lines = Path(path).read_text().splitlines()
    return np.array([[float(x) for x in l.split()[1:4]] for l in lines[2 : 2 + nwann]])


def read_sp2_sites(win, cell):
    """Fractional projection sites of the sp2 orbitals, in the order they appear in the .win."""
    sites = []
    for line in Path(win).read_text().splitlines():
        s = line.strip()
        if s.startswith("f=") and "sp2" in s:
            f = np.array([float(x) for x in s[2:].split(":")[0].split(",")])
            sites.append(f @ cell)
    return np.array(sites)


# ----------------------------------------------------------------------------- helpers
def min_image_inplane(d, cell):
    """Minimum-image in-plane displacement (small vectors only, |d| << cell)."""
    inv = np.linalg.inv(cell[:2, :2])
    f = d[..., :2] @ inv
    best = None
    for s1 in (-1, 0, 1):
        for s2 in (-1, 0, 1):
            g = f - np.round(f) + np.array([s1, s2])
            v = g @ cell[:2, :2]
            if best is None:
                best = v.copy(); shift = np.round(f) - np.array([s1, s2])
            else:
                better = np.linalg.norm(v, axis=-1) < np.linalg.norm(best, axis=-1)
                best[better] = v[better]
                shift[better] = (np.round(f) - np.array([s1, s2]))[better]
    return best, shift.astype(int)


def extract(config, indir, outdir):
    cell, pos = read_win_structure(indir / "wannier90.up.win")
    _, pos_full = read_poscar(indir / "POSCAR_novac")   # only used to locate the missing site
    nC = len(pos)
    assert len(pos_full) == nC + 1
    # missing site: the atom of POSCAR_novac with no partner in POSCAR
    dmin = [np.min(np.linalg.norm(min_image_inplane(pos - p, cell)[0], axis=1) + np.abs(pos[:, 2] - p[2]))
            for p in pos_full]
    vac = pos_full[int(np.argmax(dmin))]
    zmid = pos[:, 2].mean()
    layer = (pos[:, 2] > zmid).astype(int)  # 0 bottom, 1 top
    assert vac[2] < zmid, "vacancy must be in the bottom layer"

    # local (patch) coordinates: image of every carbon closest to the missing site
    d_ip, img = min_image_inplane(pos - vac, cell)
    local = np.c_[d_ip, pos[:, 2] - vac[2]]
    rho = np.linalg.norm(d_ip, axis=1)

    sp2_sites = read_sp2_sites(indir / "wannier90.up.win", cell)
    parents = [int(np.argmin(np.linalg.norm(min_image_inplane(pos - s, cell)[0], axis=1)
                             + np.abs(pos[:, 2] - s[2]))) for s in sp2_sites]
    sig = [nC + k for k in range(len(parents))]

    patch_pz = [int(a) for a in np.where(((layer == 0) & (rho < R_BOTTOM)) | ((layer == 1) & (rho < R_TOP)))[0]]
    # order: bottom first, then top, each by distance from the vacancy (stable, readable)
    patch_pz.sort(key=lambda a: (layer[a], round(rho[a], 3), a))
    assert all(p in patch_pz for p in parents)
    orbs = patch_pz + sig                              # Wannier orbital index of each patch orbital
    carbon = patch_pz + parents                        # carbon each patch orbital sits on
    kind = ["pz"] * len(patch_pz) + ["sigma"] * len(sig)
    parent_local = [patch_pz.index(p) for p in parents]

    result = {"config": config, "spins": {}}
    H_out = {}
    for spin in SPINS:
        R, H = read_hr(indir / f"wannier90.{spin}_hr.dat")
        i0 = int(np.where((R == 0).all(1))[0][0])
        assert np.allclose(H[i0], H[i0].conj().T, atol=1e-10)
        cen = read_centres(indir / f"wannier90.{spin}_centres.xyz", H.shape[1])
        atom_of = list(range(nC)) + parents
        # cell (relative to the POSCAR atom) in which each Wannier orbital lives
        wshift = np.round(((cen - pos[atom_of]) @ np.linalg.inv(cell))[:, :2]).astype(int)
        wpos = pos[atom_of] + np.c_[wshift, np.zeros(len(atom_of))] @ cell   # Wannier-consistent positions

        # --- step 2+3: rotation of the (pz, sp2) pairs, with gauge fixing
        U = np.eye(H.shape[1], dtype=complex)
        for p, s in zip(parents, sig):
            w, v = np.linalg.eigh(H[i0][np.ix_([p, s], [p, s])])
            v = v[:, ::-1].copy()                      # column 0 -> higher (pz), column 1 -> lower (sigma)
            for c in range(2):                         # remove arbitrary complex phases
                k = int(np.argmax(np.abs(v[:, c])))
                v[:, c] *= np.abs(v[k, c]) / v[k, c]
            if v[1, 1].real < 0:                       # sigma: sp2 coefficient positive
                v[:, 1] *= -1
            U[np.ix_([p, s], [p, s])] = v
        Hr = np.einsum("ab,rbc,cd->rad", U.conj().T, H, U)
        # pz gauge: nearest-neighbour hoppings of the rotated pz must have the bulk sign
        def nn_sum(a, Hm):
            """Sum of hoppings from orbital a to its intralayer nearest-neighbour pz orbitals."""
            tot = 0.0
            for r, Rv in enumerate(R):
                d = wpos[:nC] + np.r_[Rv @ cell[:2, :2], 0] - wpos[a]
                dist = np.linalg.norm(d, axis=1)
                nn = np.where((dist > 1.2) & (dist < 1.65))[0]
                nn = [b for b in nn if b not in parents]
                tot += sum(Hm[r, a, b].real for b in nn)
            return tot
        bulk = [a for a in range(nC) if rho[a] > R_FAR and layer[a] == 0][:5]
        bulk_sign = np.sign(sum(nn_sum(a, Hr) for a in bulk))
        flips = []
        for p in parents:
            if np.sign(nn_sum(p, Hr)) != bulk_sign:
                U[:, p] *= -1
                flips.append(p)
        Hr = np.einsum("ab,rbc,cd->rad", U.conj().T, H, U)
        for p in parents:
            assert np.sign(nn_sum(p, Hr)) == bulk_sign
        assert np.allclose(U.conj().T @ U, np.eye(len(U)), atol=1e-12)
        for p, s in zip(parents, sig):                 # rotation removed the on-site pz-sigma coupling
            assert abs(Hr[i0, p, s]) < 1e-10

        # --- step 5: matrix elements between patch orbitals, realising local bond vectors
        n = len(orbs)
        Hloc = np.zeros((n, n), complex)
        Rindex = {tuple(r): k for k, r in enumerate(R)}
        for A, (oa, ca) in enumerate(zip(orbs, carbon)):
            for B, (ob, cb) in enumerate(zip(orbs, carbon)):
                want = local[cb] - local[ca]                       # bond vector inside the patch
                have = wpos[ob] - wpos[oa]
                f = (want - have)[:2] @ np.linalg.inv(cell[:2, :2])
                Rv = np.round(f).astype(int)
                assert np.allclose(f, Rv, atol=1e-6), (A, B, f)
                Hloc[A, B] = Hr[Rindex[tuple(Rv)], oa, ob]
        assert np.allclose(Hloc, Hloc.conj().T, atol=1e-12)

        # sigma couplings that are dropped because the pz partner lies outside the patch
        out_pz = [a for a in range(nC) if a not in patch_pz]
        dropped = float(np.sqrt(sum(np.sum(np.abs(Hr[:, s, out_pz]) ** 2) for s in sig)))
        kept = float(np.sqrt(np.sum(np.abs(Hloc[len(patch_pz):, : len(patch_pz)]) ** 2)))

        # --- step 6: energy alignment per layer
        ons = np.real(np.diag(Hr[i0]))[:nC]
        far = rho >= R_FAR
        shift = {L: -float(ons[far & (layer == L)].mean()) for L in (0, 1)}
        lay = np.array([layer[c] for c in carbon])
        Hloc = Hloc + np.diag([shift[L] for L in lay])

        H_out[spin] = Hloc
        es = np.linalg.eigvalsh(Hloc[len(patch_pz):, len(patch_pz):])
        result["spins"][spin] = {
            "energy_shift_bottom_eV": shift[0], "energy_shift_top_eV": shift[1],
            "far_field_std_bottom_eV": float(ons[far & (layer == 0)].std()),
            "far_field_std_top_eV": float(ons[far & (layer == 1)].std()),
            "pz_sign_flipped_on_carbons": flips,
            "sigma_block_eigenvalues_eV": es.tolist(),
            "sigma_pi_coupling_norm_kept_eV": kept,
            "sigma_pi_coupling_norm_dropped_eV": dropped,
            "max_imag_part_eV": float(np.abs(Hloc.imag).max()),
        }

    outdir.mkdir(parents=True, exist_ok=True)
    loc = local[carbon]
    np.savez_compressed(
        outdir / "cluster.npz",
        H_up_eV=H_out["up"], H_dn_eV=H_out["dn"],
        local_positions_A=loc, kind=np.array(kind), layer=np.array([layer[c] for c in carbon]),
        sigma_parent=np.array([-1] * len(patch_pz) + parent_local),
        donor_wannier_index=np.array(orbs), donor_carbon_index=np.array(carbon),
        info=json.dumps({"config": config}),
    )
    with open(outdir / "orbitals.csv", "w") as f:
        f.write("index,kind,layer,sigma_parent,x_A,y_A,z_A,onsite_up_eV,onsite_dn_eV,donor_wannier_index\n")
        for A in range(len(orbs)):
            f.write(f"{A},{kind[A]},{'bottom' if layer[carbon[A]]==0 else 'top'},"
                    f"{(parent_local[A-len(patch_pz)] if kind[A]=='sigma' else '')},"
                    f"{loc[A,0]:.6f},{loc[A,1]:.6f},{loc[A,2]:.6f},"
                    f"{H_out['up'][A,A].real:.6f},{H_out['dn'][A,A].real:.6f},{orbs[A]}\n")
    for spin in SPINS:
        Hm = H_out[spin]
        with open(outdir / f"hoppings_{spin}.csv", "w") as f:
            f.write("a,b,distance_A,t_real_eV,t_imag_eV\n")
            for A in range(len(orbs)):
                for B in range(len(orbs)):
                    if A != B and abs(Hm[A, B]) > 1e-8:
                        dist = np.linalg.norm(loc[B] - loc[A])
                        f.write(f"{A},{B},{dist:.4f},{Hm[A,B].real:.8f},{Hm[A,B].imag:.8f}\n")
    meta = {
        "config": config,
        "n_pz": len(patch_pz), "n_sigma": len(sig), "n_orbitals": len(orbs),
        "patch_cutoffs_A": {"bottom": R_BOTTOM, "top": R_TOP, "far_field_for_energy_reference": R_FAR},
        "local_frame": "origin = missing (vacancy) carbon site of the donor; x,y,z in Angstrom; donor axes",
        "donor_cell_A": cell.tolist(),
        "donor_vacancy_site_A": vac.tolist(),
        "sigma_parent_carbons_donor": parents,
        **result,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{config}: {len(patch_pz)} pz + {len(sig)} sigma -> {outdir}")
    for spin in SPINS:
        r = result["spins"][spin]
        print(f"   {spin}: shifts bottom/top {r['energy_shift_bottom_eV']:+.4f}/{r['energy_shift_top_eV']:+.4f} eV, "
              f"sigma levels {np.round(r['sigma_block_eigenvalues_eV'],3).tolist()} eV, "
              f"sigma-pi |T| kept {r['sigma_pi_coupling_norm_kept_eV']:.3f} dropped {r['sigma_pi_coupling_norm_dropped_eV']:.3f} eV, "
              f"pz flipped {r['pz_sign_flipped_on_carbons']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", help="any of AA AB ABp (default: all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--inputs", default=str(HERE / "wannier_inputs"))
    ap.add_argument("--outputs", default=str(HERE / "clusters"))
    a = ap.parse_args()
    configs = CONFIGS if a.all or not a.configs else a.configs
    bad = [c for c in configs if c not in CONFIGS]
    if bad:
        ap.error(f"unknown config(s) {bad}; choose from {CONFIGS}")
    for c in configs:
        extract(c, Path(a.inputs) / c, Path(a.outputs) / c)


if __name__ == "__main__":
    main()
