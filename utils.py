import random
import numpy as np
import copy
import yaml
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lj import LennardJones
from pathlib import Path
from datetime import datetime
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import FIRE
from ase.calculators.eam import EAM
from ase import Atoms

with open("nn_bax_config.yml", "r") as f:
    cfg = yaml.safe_load(f)

neb_system = cfg["traj"]["NEB_system"]
N_LJ = cfg["traj"]["N_LJ"]
str_lj = "Ar" + str(N_LJ)

if neb_system == "EAM":
    POT_FILE = "Cu01.eam.alloy"                # make sure this path is correct
    ase_calculator = EAM(potential=POT_FILE, elements=["Cu"])
elif neb_system == "LJ":
    ase_calculator = LennardJones(rc=100)

def perturb_positions(positions, delta):

    perturbation = np.random.uniform(-delta, delta, positions.shape)  # Generate perturbations
    print("PERTURB: ", perturbation)
    return positions + perturbation


def pick_random_image(path, random_state): 
    # New: need random state to not just pick same im every time
    random.setstate(random_state)
    if not path:
        raise ValueError("Cannot pick from an empty path list.")

    out = random.choice(path)
    return out, random.getstate()

def random_sample_linear(linear_images_ase, delta_var):
    # Pick random index excluding first and last images
    index = random.randint(1, len(linear_images_ase) - 2)
    print("RANDOM INDEX: ",index)
    pos = perturb_positions(linear_images_ase[index].positions, delta_var)

    if neb_system == "LJ":
        atoms = Atoms(str_lj, positions=pos)
        atoms.set_cell([10, 10, 10])
        atoms.set_pbc([False, False, False])
        atoms.set_calculator(ase_calculator)

    elif neb_system == "EAM":
        cell = [[7.65796644025031, 0.0, 0.0],
                [3.828983220125155, 6.6319934785854535, 0.0],
                [0.0, 0.0, 30.252703415323648]]
        pbc = [True, True, False]

        constraint = FixAtoms(indices=[0, 1, 2, 3, 4, 5, 6, 7, 8])

        atoms = Atoms(
            symbols="Cu38",
            cell=cell,
            pbc=pbc,
            constraint=constraint,
            calculator=ase_calculator
        )
        atoms.set_positions(pos)

    e = atoms.get_potential_energy()
    f = atoms.get_forces()

    stress = np.zeros(6, dtype=float)
    spc = SinglePointCalculator(atoms, energy=float(e), forces=f, stress=stress)
    atoms.set_calculator(spc)
    return atoms
  

def add_zero_stress_spc(atoms):
    atoms_here = copy.deepcopy(atoms)
    e = atoms_here.get_potential_energy()
    f = atoms_here.get_forces()
    stress = np.zeros(6, dtype=float)
    spc = SinglePointCalculator(atoms_here, energy=float(e), forces=f, stress=stress)
    atoms_here.set_calculator(spc)
    return atoms_here


def tail(filename, n=20):
    # open with explicit utf-8 and ignore any bad bytes
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    return "\n".join(lines[-n:])



def find_latest_best_ckpt(base_dir: str) -> str:
    base = Path(base_dir)
    # find all best_checkpoint.pt files
    ckpts = list(base.rglob("best_checkpoint.pt"))
    if not ckpts:
        raise FileNotFoundError("No best_checkpoint.pt files found")

    def parse_timestamp(p: Path):
        # parent folder name, e.g. "2025-07-17-15-58-08-ft-lj7"
        folder = p.parent.name
        # split off the "-ft-..." suffix so we only parse the date part
        date_str, _, rest = folder.partition("-ft-")
        # rebuild full label (so we can return the original string)
        full_label = folder
        # parse into a datetime for comparison
        dt = datetime.strptime(date_str, "%Y-%m-%d-%H-%M-%S")
        return dt, full_label

    # build list of (datetime, label) and pick the max
    dt_labels = [parse_timestamp(p) for p in ckpts]
    latest_dt, latest_label = max(dt_labels, key=lambda x: x[0])
    return latest_label



def delete_checkpoints_in_dir(dir_path):
    """
    Delete both `checkpoint.pt` and `best_checkpoint.pt` files in the given directory.

    Parameters
    ----------
    dir_path : str or pathlib.Path
        Path to the checkpoint directory (e.g. ".../2025-07-18-15-28-16-ft-lj7").

    Returns
    -------
    None
    """
    base = Path(dir_path)
    for name in ("checkpoint.pt", "best_checkpoint.pt"):
        file = base / name
        if file.exists():
            try:
                file.unlink()
                print(f"Deleted {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        else:
            print(f"No {name} found in {base}")



def compute_sample_mae(sampled_image, equiformer_calculator_prev):
    """
    Compute the combined energy+force MAE of `equiformer_calculator_prev`
    on the given `sampled_image`.

    Parameters
    ----------
    sampled_image : ase.Atoms
        The ASE Atoms object for the newly sampled configuration.
        It should already have its “true” calculator (e.g. LJ) set so that
        get_potential_energy() and get_forces() return reference values.
    equiformer_calculator_prev : ase.Calculator
        An ASE calculator wrapping the EquiformerV2 model from the
        (i-1)-th iteration.

    Returns
    -------
    float
        Mean absolute error over [ΔE, ΔF_x, ΔF_y, ΔF_z, …].
    """
    # 1) reference (“true”) energy & forces
    sampled_image.set_calculator(ase_calculator)
    e_true = sampled_image.get_potential_energy()
    f_true = sampled_image.get_forces()

    # 2) model prediction
    sampled_image.set_calculator(equiformer_calculator_prev)
    e_pred = sampled_image.get_potential_energy()
    f_pred = sampled_image.get_forces()

    # 3) combine errors into one vector
    errs = np.concatenate((
        np.array([e_pred - e_true]),  # scalar energy error
        (f_pred - f_true).ravel()     # all force components
    ))

    # 4) MAE
    return np.mean(np.abs(errs))