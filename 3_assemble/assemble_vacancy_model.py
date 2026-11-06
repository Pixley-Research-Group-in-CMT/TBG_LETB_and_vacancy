#!/usr/bin/env python
"""Piece 3: embed a vacancy cluster (piece 2) into a pristine TBG host (piece 1).

    python assemble_vacancy_model.py --angle 1.05 --config AA --out models/theta_1.05_AA
    python assemble_vacancy_model.py --host ../1_pristine_host/hosts/pristine_theta_1.35.npz \
        --cluster ../2_vacancy_cluster/clusters/AB/cluster.npz --out models/theta_1.35_AB

Steps (README section 3):
 1. Choose the host carbon to remove: a bottom-layer site whose local stacking matches the cluster
    (AA: at the AA centre; AB: under a top-layer atom near the AB centre; ABp: under a top-layer
    hexagon centre near the AB centre), or any site given with --site.
 2. Rotate the cluster about z so its 58 (63 for ABp) pz positions fall on host carbons of the same
    layer. The best rotation is used; --jt-variant 1/2 picks one of the two other Jahn-Teller
    orientations (the isolated vacancy neighbour points along a different bond).
 3. Delete the host site. Inside the patch replace every host pz-pz hopping and on-site energy with
    the cluster (Wannier) value; keep LETB for all bonds that leave the patch.  (--pi-block letb
    keeps LETB inside the patch as well, i.e. the paper's hybridization-function construction.)
 4. Append the three sigma orbitals with their cluster couplings to the patch pz orbitals.
 5. Write one model per spin (identical orbital ordering), plus a mapping table and metadata.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tbmodel import TBModel  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAX_MISMATCH_A = 0.40   # largest allowed distance between a rotated cluster carbon and its host carbon

# Vacancy sites used for the PRL calculations, as indices of the packaged POSCAR_theta_* / pristine hosts
# (identified by coordinates from the archived vac_index.txt files). Keyed by number of host carbons.
PAPER_SITES = {
    11908: {"angle": "1.05", "AA": 6413, "AB": 1104, "ABp": 1990},
    7204: {"angle": "1.35", "AA": 3792, "AB": 5388, "ABp": 719},
    3268: {"angle": "2.00", "AA": 1334, "AB": 1046, "ABp": 1047},
}


# ----------------------------------------------------------------------------- geometry helpers
def min_image(d, cell):
    """Minimum-image in-plane displacement and the integer translation that realises it."""
    A = cell[:2, :2]
    f = d[..., :2] @ np.linalg.inv(A)
    base = np.round(f)
    best_v, best_s = None, None
    for s1 in (-1, 0, 1):
        for s2 in (-1, 0, 1):
            s = base + np.array([s1, s2])
            v = (f - s) @ A
            if best_v is None:
                best_v, best_s = v, s.copy()
            else:
                m = np.linalg.norm(v, axis=-1) < np.linalg.norm(best_v, axis=-1)
                best_v[m], best_s[m] = v[m], s[m]
    return best_v, -best_s.astype(int)   # d + S @ A = v


def layers_of(pos):
    zmid = 0.5 * (pos[:, 2].min() + pos[:, 2].max())
    return (pos[:, 2] > zmid).astype(int)


def nearest_other_layer_distance(pos, cell, layer, sites):
    """In-plane distance from each site to the nearest carbon of the other layer."""
    sites = np.asarray(sites)
    out = np.empty(len(sites))
    for L in (0, 1):
        m = layer[sites] == L
        other = pos[layer != L, :2]
        tiles = np.vstack([other + a * cell[0, :2] + b * cell[1, :2] for a in (-1, 0, 1) for b in (-1, 0, 1)])
        if m.any():
            out[m] = cKDTree(tiles).query(pos[sites[m], :2])[0]
    return out


def stacking_centres(pos, cell, layer):
    """Positions of the AA and AB centres of the moire cell."""
    bot = np.where(layer == 0)[0]
    d = nearest_other_layer_distance(pos, cell, layer, bot)
    aa_site = int(bot[np.argmin(d)])
    aa = pos[aa_site].copy()
    ab = aa + (cell[0] + 2 * cell[1]) / 3.0          # one of the two AB/BA centres
    return aa_site, aa, ab


def choose_site(pos, cell, layer, config, half_box=5.0):
    """Reproduces the sites used for the PRL (AA, AB at all angles; ABp at 1.05 and 1.35 deg)."""
    aa_site, aa, ab = stacking_centres(pos, cell, layer)
    if config == "AA":
        return aa_site, "bottom carbon with the smallest in-plane distance to a top carbon (AA centre)"
    bot = np.where(layer == 0)[0]
    v, _ = min_image(pos[bot] - ab, cell)
    near = bot[np.all(np.abs(v) < half_box, axis=1)]          # 10 A x 10 A box around the AB centre
    dist_c = np.linalg.norm(min_image(pos[near] - ab, cell)[0], axis=1)
    d_top = nearest_other_layer_distance(pos, cell, layer, near)
    if config == "AB":
        site = int(near[np.argmin(d_top)])
        return site, "bottom carbon most directly below a top carbon within a 10 A box around the AB centre AA+(a1+2a2)/3"
    want = np.abs(d_top - 1.42) < 0.4
    if not want.any():
        raise SystemExit("no bottom site below a top-layer hexagon centre near the AB centre")
    site = int(near[want][np.argmin(dist_c[want])])
    return site, "bottom carbon below a top-layer hexagon centre, closest to the AB centre AA+(a1+2a2)/3"


def rotation_matrix(phi_deg):
    c, s = np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))
    return np.array([[c, -s], [s, c]])


def map_cluster(host_pos, cell, host_layer, site, loc, loc_layer, jt_variant=0):
    """Find the rotation that maps the cluster pz positions onto host carbons."""
    v, S = min_image(host_pos - host_pos[site], cell)
    near = np.where(np.linalg.norm(v, axis=1) < 12.0)[0]
    near = near[near != site]
    trees = {L: (cKDTree(v[near[host_layer[near] == L]]), near[host_layer[near] == L]) for L in (0, 1)}

    def cost(phi):
        r = loc[:, :2] @ rotation_matrix(phi).T
        d = np.empty(len(loc))
        for L in (0, 1):
            m = loc_layer == L
            d[m] = trees[L][0].query(r[m])[0]
        return d

    coarse = np.arange(0.0, 360.0, 0.5)
    rms = np.array([np.sqrt(np.mean(cost(p) ** 2)) for p in coarse])
    # local minima, then keep the best three that are mutually > 60 degrees apart
    minima = [coarse[k] for k in range(len(coarse)) if rms[k] <= rms[k - 1] and rms[k] <= rms[(k + 1) % len(coarse)]]
    minima.sort(key=lambda p: rms[int(round(p / 0.5)) % len(coarse)])
    picked = []
    for p in minima:
        if all(min(abs(p - q), 360 - abs(p - q)) > 60 for q in picked):
            picked.append(p)
    refined = []
    for p in picked[:3]:
        fine = np.arange(p - 1.0, p + 1.0, 0.01)
        f_rms = [np.sqrt(np.mean(cost(x) ** 2)) for x in fine]
        refined.append((float(fine[int(np.argmin(f_rms))]) % 360.0, float(min(f_rms))))
    refined.sort(key=lambda x: x[1])
    if jt_variant >= len(refined):
        raise SystemExit(f"only {len(refined)} rotation(s) found")
    phi, best = refined[jt_variant]

    r = loc[:, :2] @ rotation_matrix(phi).T
    host_idx = np.empty(len(loc), int)
    mismatch = np.empty(len(loc))
    for L in (0, 1):
        m = np.where(loc_layer == L)[0]
        dd, kk = trees[L][0].query(r[m])
        host_idx[m] = trees[L][1][kk]
        mismatch[m] = dd
    if len(set(host_idx.tolist())) != len(host_idx):
        raise SystemExit("cluster-to-host mapping is not one-to-one")
    if mismatch.max() > MAX_MISMATCH_A:
        raise SystemExit(f"largest cluster/host position mismatch {mismatch.max():.3f} A > {MAX_MISMATCH_A} A: "
                         "the chosen site does not have the cluster's stacking")
    return {"phi_deg": phi, "rms_A": best, "candidates": refined, "host_index": host_idx,
            "image": S[host_idx], "mismatch_A": mismatch}


# ----------------------------------------------------------------------------- assembly
def assemble(host, cl, spin, site, mp, pi_block="wannier", with_sigma=True):
    n_host = host.norb
    kind = np.asarray(cl["kind"]).astype(str)
    pz = np.where(kind == "pz")[0]
    sg = np.where(kind == "sigma")[0]
    Hc = cl[f"H_{spin}_eV"]
    hidx = np.empty(len(kind), int)
    img = np.empty((len(kind), 2), int)
    hidx[pz], img[pz] = mp["host_index"], mp["image"]
    parent = np.asarray(cl["sigma_parent"])
    hidx[sg], img[sg] = hidx[parent[sg]], img[parent[sg]]

    in_patch = np.zeros(n_host, bool)
    in_patch[hidx[pz]] = True
    patch_image = np.zeros((n_host, 2), int)
    patch_image[hidx[pz]] = img[pz]

    i, j, R, t = host.i, host.j, host.R, host.t
    keep = (i != site) & (j != site)
    both = in_patch[i] & in_patch[j]
    local = both & np.all(R[:, :2] == patch_image[j] - patch_image[i], axis=1)
    assert not np.any(both & ~local), "host has a long-range hopping between two patch carbons"
    onsite = host.onsite.copy()
    if pi_block == "wannier":
        keep &= ~local
        onsite[hidx[pz]] = np.real(np.diag(Hc)[pz])

    # new numbering: drop the vacancy, append sigma orbitals at the end
    newidx = np.arange(n_host) - (np.arange(n_host) > site)
    newidx[site] = -1
    I, J, RR, T = [newidx[i[keep]]], [newidx[j[keep]]], [R[keep]], [t[keep]]
    n_pi = n_host - 1
    orb_new = {}
    for A in pz:
        orb_new[A] = newidx[hidx[A]]
    if with_sigma:
        for k, A in enumerate(sg):
            orb_new[A] = n_pi + k

    def add(A, B):
        Rv = np.r_[img[B] - img[A], 0]
        I.append([orb_new[A]]); J.append([orb_new[B]]); RR.append([Rv]); T.append([Hc[A, B]])

    if pi_block == "wannier":
        for A in pz:
            for B in pz:
                if A != B and Hc[A, B] != 0:
                    add(A, B)
    if with_sigma:
        for A in sg:
            for B in list(pz) + list(sg):
                if A != B and Hc[A, B] != 0:
                    add(A, B)
                    add(B, A)

    pos = np.delete(host.positions, site, axis=0)
    kinds = ["pz"] * n_pi
    if with_sigma:
        onsite = np.r_[np.delete(onsite, site), np.real(np.diag(Hc)[sg])]
        pos = np.vstack([pos, host.positions[hidx[sg]]])
        kinds += ["sigma"] * len(sg)
    else:
        onsite = np.delete(onsite, site)
    I, J, RR, T = (np.concatenate(x) for x in (I, J, RR, T))
    # coalesce (sigma-sigma pairs were added from both ends)
    key = np.stack([I, J, RR[:, 0], RR[:, 1]], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    first = np.full(len(uniq), -1)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    T = T[first]
    I, J, RR = uniq[:, 0], uniq[:, 1], np.c_[uniq[:, 2:4], np.zeros(len(uniq), int)]
    pristine = np.r_[np.delete(np.arange(n_host), site), hidx[sg] if with_sigma else []].astype(int)
    m = TBModel(I, J, RR, T, onsite, pos, host.cell,
                orbital_kind=np.array(kinds), pristine_site=pristine)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--angle", choices=["1.05", "1.35", "2.00"], help="use the packaged host for this angle")
    ap.add_argument("--config", choices=["AA", "AB", "ABp"], help="use the packaged cluster for this stacking")
    ap.add_argument("--host", help="pristine host .npz (overrides --angle)")
    ap.add_argument("--cluster", help="cluster.npz (overrides --config)")
    ap.add_argument("--site", default="paper",
                    help="host carbon to remove: 'paper' (default; the PRL site, packaged hosts only), "
                         "'auto' (stacking rule), or a 0-based index")
    ap.add_argument("--supercell", type=int, default=1, metavar="N",
                    help="tile the host N x N before inserting ONE vacancy (isolated vacancy, e.g. for KPM); "
                         "the 'paper' site then refers to the home copy of the packaged host")
    ap.add_argument("--jt-variant", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--pi-block", choices=["wannier", "letb"], default="wannier")
    ap.add_argument("--no-sigma", action="store_true", help="do not add the three sigma orbitals")
    ap.add_argument("--spins", nargs="+", default=["up", "dn"], choices=["up", "dn"])
    ap.add_argument("--csv", action="store_true", help="also write CSV copies of the models")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()

    host_path = Path(a.host) if a.host else ROOT / f"1_pristine_host/hosts/pristine_theta_{a.angle}.npz"
    cl_path = Path(a.cluster) if a.cluster else ROOT / f"2_vacancy_cluster/clusters/{a.config}/cluster.npz"
    host = TBModel.load(host_path)
    n_prim = host.norb
    if a.supercell > 1:
        host = host.supercell(a.supercell, a.supercell)
    cl = np.load(cl_path)
    config = json.loads(str(cl["info"]))["config"]
    hlayer = layers_of(host.positions)

    if a.site == "paper" and n_prim in PAPER_SITES:
        site = PAPER_SITES[n_prim][config]          # home copy (0,0) keeps the primitive indices
        rule = f"site used in the PRL for theta={PAPER_SITES[n_prim]['angle']} deg, {config}"
    elif a.site in ("paper", "auto"):
        if a.site == "paper":
            print("host not one of the packaged hosts: falling back to --site auto")
        site, rule = choose_site(host.positions, host.cell, hlayer, config)
    else:
        site, rule = int(a.site), "given on the command line"
    if hlayer[site] != 0:
        raise SystemExit("the vacancy must be on the bottom layer (the cluster was computed that way)")
    kind = np.asarray(cl["kind"]).astype(str)
    pz = kind == "pz"
    mp = map_cluster(host.positions, host.cell, hlayer, site,
                     cl["local_positions_A"][pz], cl["layer"][pz], a.jt_variant)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spin in a.spins:
        m = assemble(host, cl, spin, site, mp, a.pi_block, not a.no_sigma)
        herm = m.hermiticity_error()
        assert herm < 1e-10, herm
        m.info = {"spin": spin, "config": config, "host": host_path.name, "host_supercell": a.supercell,
                  "removed_host_site": site,
                  "pi_block": a.pi_block, "sigma_orbitals": not a.no_sigma,
                  "convention": "H_ij(k)=onsite_i delta_ij + sum_R t_ij(R) exp(2 pi i k.R); both directions stored"}
        m.save(out / f"model_{spin}.npz")
        if a.csv:
            m.to_csv(str(out / f"model_{spin}"))
        print(f"{out}/model_{spin}.npz: {m.norb} orbitals ({(m.extra['orbital_kind']=='sigma').sum()} sigma), "
              f"{len(m.t)} directed hoppings, hermiticity error {herm:.1e}")

    with open(out / "cluster_mapping.csv", "w") as f:
        f.write("cluster_orbital,kind,layer,host_site,new_orbital,image_n1,image_n2,mismatch_A\n")
        pz_ids = np.where(pz)[0]
        for k, A in enumerate(pz_ids):
            h = mp["host_index"][k]
            f.write(f"{A},pz,{cl['layer'][A]},{h},{h - (h > site)},{mp['image'][k][0]},{mp['image'][k][1]},"
                    f"{mp['mismatch_A'][k]:.4f}\n")
    n_pi = host.norb - 1
    meta = {
        "config": config, "host": str(host_path), "host_supercell": [a.supercell, a.supercell],
        "cluster": str(cl_path),
        "removed_host_site": int(site), "removed_site_position_A": host.positions[site].tolist(),
        "site_rule": rule, "rotation_deg": mp["phi_deg"], "jt_variant": a.jt_variant,
        "rotation_candidates_deg_rms_A": mp["candidates"],
        "position_mismatch_A": {"rms": float(np.sqrt(np.mean(mp["mismatch_A"] ** 2))), "max": float(mp["mismatch_A"].max())},
        "pi_block": a.pi_block, "sigma_orbitals": not a.no_sigma,
        "orbitals": {"pz": [0, n_pi - 1], "sigma": [n_pi, n_pi + 2] if not a.no_sigma else None},
        "neutral_filling": "total electron number (both spins) = number of orbitals per spin model",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"removed host site {site} ({rule}); rotation {mp['phi_deg']:.2f} deg, "
          f"position mismatch rms {meta['position_mismatch_A']['rms']:.3f} / max {meta['position_mismatch_A']['max']:.3f} A")
    print("other Jahn-Teller orientations:", [(round(p, 2), round(r, 3)) for p, r in mp["candidates"]])


if __name__ == "__main__":
    main()
