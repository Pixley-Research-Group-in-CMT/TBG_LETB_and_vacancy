# Carbon vacancy in twisted bilayer graphene: tight-binding models

This package provides the atomistic tight-binding ingredients behind
Y. Chang *et al.*, *Vacancy-Induced Tunable Kondo Effect in Twisted Bilayer Graphene*,
[PRL **133**, 126503 (2024)](https://doi.org/10.1103/PhysRevLett.133.126503)
([arXiv:2312.09286](https://arxiv.org/abs/2312.09286)), and a script that combines them into a single
tight-binding model of TBG with one vacancy.

It has three pieces:

| | Folder | What it is |
|---|---|---|
| 1 | [`1_pristine_host/`](1_pristine_host/) | Relaxed TBG structures at 1.05°, 1.35° and 2.0°. The pristine p<sub>z</sub> tight-binding host built from them with a slightly patched LETB model. |
| 2 | [`2_vacancy_cluster/`](2_vacancy_cluster/) | The vacancy cluster: a 61-orbital local Hamiltonian from DFT + Wannier functions for a bottom-layer vacancy in untwisted bilayer graphene. It comes in three local stackings (AA, AB, AB′) and both spins. |
| 3 | [`3_assemble/`](3_assemble/) | `assemble_vacancy_model.py` inserts piece 2 into piece 1 and writes the final model. `check_model.py` checks the result. |

**Data provenance.** All three clusters come from the *re-relaxed* bilayer Wannier runs used for the paper (6×6×1 *k*-grid relaxation, Sept 2023; on Amarel in `/scratch/yc1329/bilayer_grph/{AA,AB,AB_2}_rerelaxed/sc6_k3x3_struct3/`). They were checked against the paper-era archive:

- The AA/AB geometries are identical to `re-relax_struc_k661/CONTCAR`.
- The σ–π couplings equal the archived Sept-2023 hybridization tables to 1e-15 eV.
- The Γ-point spectra match the archived band data behind Table S1 / Fig. S3 to within 7 meV.

An earlier pre-relaxation set (May 2023) also exists in the research archive. **Do not use it**: its AA σ–π coupling is about twice the paper's.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, scipy: all you need for pieces 2-3

cd 3_assemble
python assemble_vacancy_model.py --angle 1.05 --config AA --out models/theta_1.05_AA
python check_model.py models/theta_1.05_AA
```

```python
import sys; sys.path.insert(0, "3_assemble")
from tbmodel import TBModel
m = TBModel.load("3_assemble/models/theta_1.05_AA/model_up.npz")
Hk = m.bloch_hamiltonian(k=(1/3, 1/3))          # scipy.sparse CSR, eV
Hr = m.realspace_supercell(4, 4, periodic=True)  # 4x4 moiré cells (contains 16 vacancies!)
```

For a single, isolated vacancy (e.g. KPM local DOS), tile the pristine host first and insert one vacancy:

```bash
python assemble_vacancy_model.py --angle 1.05 --config AB --supercell 3 --out models/theta_1.05_AB_3x3
```

```python
m = TBModel.load("3_assemble/models/theta_1.05_AB_3x3/model_up.npz")
H = m.bloch_hamiltonian((0, 0))   # sparse, ~1e5 orbitals; feed to KPM (rescale by its spectral bounds)
sigma = range(m.norb - 3, m.norb)   # sigma orbitals are always the last three
```

Assembling takes about a second. Only rebuilding the pristine host (piece 1) needs LETB.

---

## File format of every model (`.npz`)

| key | shape | meaning |
|---|---|---|
| `i`, `j` | (m,) | orbital indices of a directed hopping t<sub>ij</sub>(**R**) = ⟨0,i\|H\|**R**,j⟩ |
| `R` | (m,3) | integer translation in units of the rows of `cell_A` (`R[:,2]=0`) |
| `t_eV` | (m,) complex | hopping |
| `onsite_eV` | (n,) | on-site energies |
| `positions_A` | (n,3) | orbital positions in the home cell (Å) |
| `cell_A` | (3,3) | lattice vectors as rows (Å); a₁=(L,0,0), a₂=(−L/2, L√3/2, 0) |
| `orbital_kind` | (n,) | `pz` or `sigma` (assembled models only) |
| `pristine_site` | (n,) | index of the orbital's carbon in the pristine host (assembled models only) |

H<sub>ij</sub>(**k**) = ε<sub>i</sub>δ<sub>ij</sub> + Σ<sub>**R**</sub> t<sub>ij</sub>(**R**) e<sup>2πi **k**·**R**</sup>, with **k** in reciprocal-lattice fractions.
**Both (i,j,R) and (j,i,−R) are stored. Do not add the Hermitian conjugate again.**
For these cells the moiré K point is **k** = (1/3, 1/3).

---

## 1. Pristine host: LETB + patch

**Model.** The host is the machine-learned "LETB" tight-binding model of
[Pathak *et al.*, PRB 105, 115141 (2022)](https://doi.org/10.1103/PhysRevB.105.115141)
([code](https://github.com/qmc-hamm/bilayer_tight_binding), [PyPI `bilayer-letb`](https://pypi.org/project/bilayer-letb/)).
It has one p<sub>z</sub> orbital per carbon and environment-dependent intra- and interlayer hoppings, with on-site energies of 0.

**Structures.** [`structures/POSCAR_theta_{1.05,1.35,2.00}`](1_pristine_host/structures/) contain one moiré cell each, with 11908, 7204 and 3268 C atoms. They were relaxed with LAMMPS (REBO intralayer + KC interlayer, as in the paper). They are byte-identical to the files used on Amarel for the paper.

**Prebuilt hosts.** [`hosts/pristine_theta_*.npz`](1_pristine_host/hosts/) were built from these structures with the patched LETB.

**The patch.** [`letb_patch/letb_vacancy_windows.patch`](1_pristine_host/letb_patch/letb_vacancy_windows.patch) changes one function, `partition_tb` in `descriptors/descriptors_intralayer.py`. That function sorts intralayer pairs into first/second/third-neighbour shells by distance. The public release only accepts bonds within ±5% of the ideal lengths and sets all other hoppings to zero. Around a relaxed vacancy, bonds distort by more than that and would be dropped. The patch widens the windows:

| shell | public 0.0.2 | patched |
|---|---|---|
| 1st | [0.95, 1.05] d<sub>min</sub> | [0.80, 1.20] d<sub>min</sub> |
| 2nd | [0.95, 1.05] √3 d<sub>min</sub> | [0.90, 1.10] √3 d<sub>min</sub> |
| 3rd | [0.95, 1.05] 2 d<sub>min</sub> | unchanged |

The fitted parameters are unchanged; this is not a refit.

**For pristine TBG the patch has no effect.** The 2.0° host built with and without it is bit-identical, because no pristine bond is distorted by 5%. The patch only matters if you run LETB directly on a defect geometry. This package doesn't do that: the defect region comes from DFT instead (piece 2).

**Rebuilding a host:**

```bash
pip install "numpy<2" scipy ase pythtb==1.8.0
pip download --no-deps --no-binary :all: bilayer-letb==0.0.2 && tar xzf bilayer_letb-0.0.2.tar.gz
cd bilayer_letb-0.0.2 && patch -p1 < ../1_pristine_host/letb_patch/letb_vacancy_windows.patch && pip install --no-deps . && cd ..
python 1_pristine_host/build_pristine_host.py 1_pristine_host/structures/POSCAR_theta_1.05 my_host.npz
```

The 1.05° build takes about 3 min and about 10 GB of RAM. The script refuses to run on an unpatched install unless you pass `--allow-unpatched`.

**Results.** The pristine hosts reproduce the expected physics:

- Charge neutrality is the four-fold Dirac point at K: E<sub>CNP</sub> = −0.6835 eV (1.05°), −0.6851 eV (1.35°), −0.6876 eV (2.0°). It is not 0, because of LETB's second-neighbour hopping.
- At 1.05° the four central bands spread over 16 meV.

---

## 2. Vacancy cluster

**Source.** The cluster comes from a spin-polarized DFT (VASP, PBE) calculation: a 6×6 bilayer-graphene supercell with one bottom-layer vacancy, 143 C atoms. It was fully relaxed with a 6×6×1 *k*-grid, including the Jahn–Teller distortion, in which one neighbour of the vacancy moves away and the other two re-bond.

**Downfolding.** The DFT bands were downfolded with Wannier90 to 146 orbitals: 143 p<sub>z</sub> plus one sp² dangling bond on each of the three neighbours of the vacancy. The 3×3 *k*-mesh uses a disentanglement window of −25…16 eV and a frozen window of −5…1 eV (AB′: −4.5…0 eV).

**Files.** The raw Wannier90 files are in [`wannier_inputs/<config>/`](2_vacancy_cluster/wannier_inputs/), for AA, AB and ABp (= AB′):

- `wannier90.{up,dn}_hr.dat.gz`, `*_centres.xyz`, `*.win`
- `POSCAR`: the relaxed geometry, copied from the `.win`
- `POSCAR_novac`: the ideal lattice, used only to locate the missing atom

**Stackings:**

- **AA:** the vacancy has a top-layer atom directly above it.
- **AB:** Bernal stacking, with a top-layer atom above the vacancy.
- **AB′:** Bernal stacking, with the vacancy below a top-layer hexagon centre.

In TBG, AA vacancies sit in the AA regions; AB and AB′ vacancies sit in the AB/BA domains.

### What `extract_cluster.py` does

1. **Separate σ from p<sub>z</sub>.** On each of the three neighbour carbons, the p<sub>z</sub> and sp² Wannier functions are mixed. Each on-site 2×2 block is diagonalized. The higher level is called p<sub>z</sub> and the lower one **σ**, so the on-site p<sub>z</sub>–σ coupling becomes zero.
2. **Fix the gauge.** Each rotated p<sub>z</sub> gets the sign that makes its nearest-neighbour hoppings match the bulk sign. The Wannier p<sub>z</sub> orbitals then share LETB's sign convention, as checked bond by bond against LETB on the same geometry. Each σ is fixed by making its sp² coefficient positive.
3. **Select the patch.** It keeps bottom-layer carbons within 6 Å and top-layer carbons within 4 Å (in plane) of the missing atom: 39 + 19 = 58 p<sub>z</sub> (39 + 24 = 63 for AB′), plus 3 σ. No coordination shell lies near either cutoff.
4. **Write a finite matrix.** Every matrix element between patch orbitals is written into one dense matrix of size 61×61 (66×66 for AB′), in a local frame with the missing atom at the origin. The cell translation for each element is chosen from the Wannier centres so that it realizes the *local* bond. This matters because some Wannier orbitals sit one lattice vector away from their atom in the POSCAR.
5. **Set the energy reference.** Each layer is shifted so that the average p<sub>z</sub> on-site energy far from the vacancy (≥ 5 Å in plane) is 0, LETB's on-site value. The σ orbitals get the bottom-layer shift.

### Outputs in `clusters/<config>/`

| file | content |
|---|---|
| `cluster.npz` | `H_up_eV`, `H_dn_eV` (local Hamiltonians); `local_positions_A`; `kind`; `layer`; `sigma_parent` |
| `orbitals.csv` | index, kind, layer, position, on-site energy for each spin |
| `hoppings_{up,dn}.csv` | every nonzero element with its bond length |
| `metadata.json` | energy shifts, σ-block levels, coupling norms (kept vs. dropped), sign flips |

To regenerate them: `python extract_cluster.py --all` (NumPy only, about 2 min).

### Physical content

- **σ block.** The 3×3 σ block has a bonding level about 6.6 eV below the p<sub>z</sub> on-site energy, from the two re-bonded atoms. Its **middle eigenmode is the V<sub>σ</sub> local moment** of the paper. The third level is antibonding.
- **Spin splitting.** The two spins differ mainly in that V<sub>σ</sub> level: about −1.1 eV for ↑ and +1.1 eV for ↓, relative to the far-field p<sub>z</sub> on-site energy.
- **Couplings dropped by the patch cutoff.** σ couplings to p<sub>z</sub> orbitals outside the patch are left out. They are 0.02–0.03 eV in norm, against 0.2–0.3 eV (↑) and 0.8–1.4 eV (↓) kept.

The V<sub>π</sub> "zero mode" isn't a separate orbital. It lives in the reconstructed p<sub>z</sub> block and emerges once the cluster is embedded.

---

## 3. Assembling the vacancy model

```bash
python assemble_vacancy_model.py --angle {1.05,1.35,2.00} --config {AA,AB,ABp} --out DIR [options]
```

### Steps

1. **Pick the vacancy site.** By default (`--site paper`) the script uses the bottom-layer carbon the paper used; these were identified by coordinates from the archived calculations. Alternatives:
   - `--site auto` applies the stacking rule. AA: the bottom carbon closest to a top carbon. AB: the bottom carbon most directly below a top carbon near the AB centre AA+(a₁+2a₂)/3. AB′: the bottom carbon below a top hexagon centre, closest to that centre.
   - `--site N` uses any bottom-layer carbon you choose.
2. **Place the cluster.** The cluster is rotated about z so that its p<sub>z</sub> sites land on host carbons of the same layer. The largest position mismatch is 0.25–0.27 Å in all nine angle × stacking combinations, at the Jahn–Teller-displaced neighbours of the vacancy; the r.m.s. mismatch is below 0.1 Å. The script refuses if the mismatch exceeds 0.4 Å or the mapping isn't one-to-one.
   - **Jahn–Teller orientation.** Three orientations are nearly equivalent, depending on which neighbour bond the isolated atom points along. `--jt-variant 1|2` selects the others.
3. **Combine the Hamiltonians.**
   - Delete the host carbon at the vacancy site.
   - **Inside the patch**, replace all p<sub>z</sub>–p<sub>z</sub> hoppings and on-site energies with the cluster values.
   - **Bonds that cross the patch boundary**, and everything outside, keep LETB.
4. **Add σ orbitals.** The three σ orbitals become the last three orbitals. They carry their on-site block and couplings to the patch p<sub>z</sub>. Their positions are set to their parent carbons.
5. **Write the output.** `DIR/model_up.npz` and `model_dn.npz` have identical orbital order: p<sub>z</sub> 0 … N−2, then σ N−1 … N+1. The folder also gets `cluster_mapping.csv` (cluster orbital → host carbon → new index, with the mismatch) and `metadata.json`.

### Options

| option | effect |
|---|---|
| `--pi-block letb` | Keep LETB inside the patch: just delete the site and attach σ. This is the construction behind the paper's hybridization function, where pristine TBG eigenstates are coupled to V<sub>σ</sub>. |
| `--no-sigma` | p<sub>z</sub>-only reconstructed vacancy |
| `--spins up` | build only one spin |
| `--csv` | also write CSV copies |
| `--host`, `--cluster` | use your own files (e.g. a larger pristine host) |
| `--supercell N` | tile the host N×N, then insert **one** vacancy into the home copy (isolated vacancy; `--site paper` still works) |

### Using the models

- **Periodic vacancies.** The model is **one moiré cell with one vacancy**. Bloch Hamiltonians and `realspace_supercell` therefore describe a vacancy *lattice*: the n₁×n₂ supercell holds n₁n₂ vacancies. An isolated vacancy needs a large pristine host with a single cluster inserted: use `--supercell N`, which tiles the host with `TBModel.supercell` and puts one vacancy in it. You can also pass your own larger host with `--host`. A 1.05° 2×2 model has 47 634 orbitals and 1.5 M hoppings and is built in 3 s. In a 2.0° 3×3 model the 8 Å neighbourhood of the vacancy is identical to that of the single-cell model, and V<sub>σ</sub> sits at the same energy (−0.447 eV ↑, splitting 2.218 eV).
- **Filling.** A neutral vacancy has total electron number (both spins) equal to the number of orbitals in one spin model, N + 2. Compute E<sub>F</sub> for the system you build. For the 2.0° single-vacancy cell, E<sub>F</sub> lies within 10 meV of the pristine E<sub>CNP</sub>.
- **Spin.** The cluster blocks are spin-polarized DFT one-body Hamiltonians: the exchange splitting of V<sub>σ</sub> is already inside them. Adding a Hubbard U on top (e.g. for an Anderson or BdG treatment) needs a double-counting choice. One simple non-magnetic starting point is the spin average of `H_up_eV` and `H_dn_eV`; both share the same gauge conventions after step 2 of the extraction.
- **Basis.** A pairing field for BdG must be written in this same carbon-orbital basis. Normal blocks are `model_up` for spin ↑ and `model_dn` for spin ↓.

### What `check_model.py` reports (1.05°, AA)

```
pristine host: 11908 orbitals, E_CNP = -0.6835 eV, central-band spread (Gamma,K,M) = 16.2 meV
  up: 11910 orbitals, hermiticity error 4.4e-16; ... V_sigma weight peak at -0.453 eV
  dn: 11910 orbitals, hermiticity error 2.2e-16; ... V_sigma weight peak at +1.768 eV
  V_sigma spin splitting (dn - up) = 2.221 eV
```

Across all nine models the V<sub>σ</sub>↑ level is at −0.42 to −0.46 eV from E<sub>CNP</sub>, and the spin splitting is 2.20–2.24 eV. This is the paper's Table S1 (ε ≈ −0.45 to −0.51 eV, U ≈ 2.21–2.26 eV).

"Weight captured" is the fraction of V<sub>σ</sub> spectral weight found among the 60 eigenstates nearest the level. It can be small for the strongly hybridized ↓ state; that is not an error.

---

## Known approximations

- **Patch boundary.** Far from the vacancy, LETB and the Wannier model agree to a few hundredths of an eV. In magnitude, the first-neighbour hopping is 2.91 vs 2.87 eV, the second-neighbour 0.22 vs 0.24 eV, and the vertical interlayer 0.24 vs 0.19–0.21 eV. So the patch boundary carries a mismatch of that size. The two models share the same sign convention except for the three rotated p<sub>z</sub> orbitals, which `extract_cluster.py` fixes.
- **Twist.** The cluster comes from *untwisted* bilayers. Near the AA and AB centres at θ ≤ 2°, the local registry within 6 Å is nearly that of the untwisted bilayer. The cluster is the same at all angles.
- **Corrugation.** The vacancy geometry is the free-standing DFT equilibrium. The substrate-induced corrugation u⊥ of Fig. 1(f,g), which raises V, is not included here.
- **AB′ hybridization tables.** For AB′, the paper's archived TBG hybridization tables (Sept 2023) were computed from an earlier AB′ relaxation. Its geometry differs by up to 0.33 Å; compare the two nearly degenerate AB′ distortions in SM Fig. S2. This package uses the re-relaxed AB′, the one behind Table S1 and the corrugation series of Fig. 1g / Fig. S9.

## Requirements

- NumPy and SciPy, for reading, assembling and checking.
- `bilayer-letb` 0.0.2 + patch, ASE and PythTB, only to rebuild hosts.

Please cite the PRL above and Pathak *et al.* (PRB 105, 115141) when using these models.
