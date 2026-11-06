#!/usr/bin/env python
"""Piece 1: build the pristine TBG pz tight-binding host with the (patched) LETB.

    python build_pristine_host.py structures/POSCAR_theta_1.05 hosts/pristine_theta_1.05.npz

Requires `bilayer-letb==0.0.2` with `letb_patch/letb_vacancy_windows.patch` applied
(see README.md, section 1). The script refuses to run on an unpatched install unless
--allow-unpatched is given (useful only to demonstrate that the patch does not change
the pristine host).

Output (.npz, energies in eV, lengths in Angstrom):
    i, j        int   directed hopping  t_ij(R) = <0,i|H|R,j>
    R           int   (m,3) lattice translation in units of the rows of cell_A
    t_eV        complex
    onsite_eV   (n,)  all zero for LETB
    positions_A (n,3) wrapped Cartesian positions, same order as the POSCAR
    cell_A      (3,3) rows are lattice vectors
Both directions (i,j,R) and (j,i,-R) are stored explicitly: H_ij(k) = sum_R t_ij(R) e^{2 pi i k.R}.
Do NOT add a Hermitian conjugate again.
"""
import argparse
import inspect
import json
import time
from io import StringIO
from pathlib import Path

import numpy as np
from ase.io import read


def read_poscar(path):
    """Read only the coordinate block (some archived POSCARs carry stale velocity blocks)."""
    lines = Path(path).read_text().splitlines()
    n = sum(map(int, lines[6].split()))
    start = 9 if lines[7].strip().lower().startswith("s") else 8
    return read(StringIO("\n".join(lines[: start + n]) + "\n"), format="vasp")


def letb_is_patched():
    from bilayer_letb.descriptors import descriptors_intralayer as d

    src = inspect.getsource(d.partition_tb)
    return "0.80 * min_distance" in src


def letb_to_directed(model):
    """PythTB stores each LETB entry once and adds its Hermitian conjugate in _gen_ham.
    Convert to an explicit directed list, coalescing duplicate (i,j,R) keys exactly."""
    n = model._norb
    raw = model._hoppings
    i = np.array([x[1] for x in raw], dtype=np.int64)
    j = np.array([x[2] for x in raw], dtype=np.int64)
    r = np.array([x[3] for x in raw], dtype=np.int64)
    t = np.array([x[0] for x in raw], dtype=complex)
    assert np.all(r[:, 2] == 0) and np.max(np.abs(r)) <= 1, "unexpected translation range"

    def key(i, j, r):
        return (((r[:, 0] + 1) * 3 + r[:, 1] + 1) * n + i) * n + j

    keys = np.concatenate([key(i, j, r), key(j, i, -r)])
    vals = np.concatenate([t, t.conj()])
    order = np.argsort(keys, kind="stable")
    uniq, first = np.unique(keys[order], return_index=True)
    values = np.add.reduceat(vals[order], first)
    keep = values != 0
    uniq, values = uniq[keep], values[keep]
    jj = uniq % n
    ii = (uniq // n) % n
    cell = uniq // (n * n)
    R = np.column_stack([cell // 3 - 1, cell % 3 - 1, np.zeros(len(cell), int)])
    # Hermiticity of the explicit list
    rev = key(jj, ii, -R)
    where = np.searchsorted(uniq, rev)
    assert np.array_equal(uniq[where], rev)
    herm = float(np.max(np.abs(values - values[where].conj())))
    assert herm < 1e-12, herm
    return ii, jj, R, values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poscar")
    ap.add_argument("output")
    ap.add_argument("--allow-unpatched", action="store_true")
    args = ap.parse_args()

    import bilayer_letb
    from bilayer_letb.api import tb_model

    patched = letb_is_patched()
    if not patched and not args.allow_unpatched:
        raise SystemExit("bilayer_letb is not patched (see README section 1); use --allow-unpatched to override")

    t0 = time.time()
    atoms = read_poscar(args.poscar)
    print(f"{args.poscar}: {len(atoms)} carbons, building LETB (patched={patched}) ...", flush=True)
    model = tb_model(atoms)
    i, j, R, t = letb_to_directed(model)
    onsite = np.asarray(model._site_energies, dtype=float)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        i=i, j=j, R=R, t_eV=t, onsite_eV=onsite,
        positions_A=atoms.get_positions(wrap=True), cell_A=atoms.cell.array,
        info=json.dumps({
            "kind": "pristine TBG host, one pz orbital per carbon (spin independent)",
            "source_poscar": Path(args.poscar).name,
            "letb_patched": patched,
            "letb_module": bilayer_letb.__file__,
            "convention": "H_ij(k)=sum_R t_ij(R) exp(2 pi i k.R); both directions stored; do not add h.c.",
        }),
    )
    print(f"wrote {out}: {len(onsite)} orbitals, {len(t)} directed hoppings, {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
