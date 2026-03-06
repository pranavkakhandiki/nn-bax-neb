import copy
import yaml
import numpy as np
from numpy.linalg import norm
from ase.neb import NEB
from ase.optimize import FIRE
from ase.io import Trajectory
from ase.calculators.lj import LennardJones
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import FIRE
from ase.calculators.eam import EAM


with open("nn_bax_config.yml", "r") as f:
    cfg = yaml.safe_load(f)

# Pull variables from config
traj_dir = cfg["traj"]["traj_dir"]
N_LJ = cfg["traj"]["N_LJ"]
str_lj = "Ar" + str(N_LJ)
initial_traj_num = cfg["traj"]["initial_traj_num"]
final_traj_num = cfg["traj"]["final_traj_num"]
neb_system = cfg["traj"]["NEB_system"]
remove_rot = cfg["traj"]["remove_rot"]
neb_method = cfg["traj"]["neb_method"]
apply_constraint = cfg["traj"]["apply_constraint"]
set_tags = cfg["set_tags"]

if neb_system == "EAM":
    POT_FILE = "Cu01.eam.alloy"                # make sure this path is correct
    ase_calculator = EAM(potential=POT_FILE, elements=["Cu"])
elif neb_system == "LJ":
    ase_calculator = LennardJones(rc=100)

test = Trajectory(traj_dir)

initial_ase = copy.deepcopy(test[initial_traj_num])
initial_ase.set_calculator(copy.deepcopy(ase_calculator))
final_ase = copy.deepcopy(test[final_traj_num])
final_ase.set_calculator(copy.deepcopy(ase_calculator))

def run_neb(calc, n_images, fmax_thresh, neb_steps):
    # 2.A. NEB using ASE
    n_images = n_images
    images_ase = [copy.deepcopy(initial_ase)]
    for i in range(1, n_images-1):
        image_ase = copy.deepcopy(initial_ase)
        if set_tags:
            image_ase.set_tags(np.ones(len(image_ase)))

        image_ase.set_calculator(copy.deepcopy(calc))
        images_ase.append(image_ase)
    images_ase.append(copy.deepcopy(final_ase))

    neb_ase = NEB(images_ase, climb=False, remove_rotation_and_translation=remove_rot, method= neb_method)
    neb_ase.interpolate(method='linear', apply_constraint = apply_constraint)
    qn_ase = FIRE(neb_ase)
    qn_ase.run(fmax=fmax_thresh, steps=neb_steps)

 
    
    # Extract final fmax from the NEB band
    forces = neb_ase.get_forces()
    final_fmax = norm(forces, axis=1).max()

    # Compute per‑image potentials and barrier
    energies = [img.get_potential_energy() for img in images_ase]
    energy_barrier = max(energies) - energies[0]

    return images_ase, final_fmax, energy_barrier


def run_neb_patience(calc, n_images, fmax_thresh, patience, neb_steps):
    # 2.A. NEB using ASE
    images_ase = [copy.deepcopy(initial_ase)]
    for i in range(1, n_images-1):
        image_ase = copy.deepcopy(initial_ase)
        if set_tags:
            image_ase.set_tags(np.ones(len(image_ase)))
        image_ase.set_calculator(copy.deepcopy(calc))
        images_ase.append(image_ase)
    images_ase.append(copy.deepcopy(final_ase))

    neb_ase = NEB(images_ase, climb=False, remove_rotation_and_translation=remove_rot, method=neb_method)
    neb_ase.interpolate(method='linear', apply_constraint = apply_constraint)
    qn_ase = FIRE(neb_ase)

    best_fmax = np.inf
    no_improve = 0

    for step in range(neb_steps):
        qn_ase.step()

        # Compute current fmax
        forces = neb_ase.get_forces()
        fmax = norm(forces, axis=1).max()

        # Compute current energies
        energies = [img.get_potential_energy() for img in images_ase]

        # Print progress
        print(f"Step {step:4d} | fmax = {fmax:.6f} | max(E) = {max(energies):.6f} | min(E) = {min(energies):.6f}")

        # Check threshold
        if fmax < fmax_thresh:
            print("Stopping: reached fmax threshold")
            break

        # Check patience
        if fmax < best_fmax:
            best_fmax = fmax
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Stopping: no improvement in fmax for {patience} steps")
            break

    # Final evaluation
    forces = neb_ase.get_forces()
    final_fmax = norm(forces, axis=1).max()
    energies = [img.get_potential_energy() for img in images_ase]
    energy_barrier = max(energies) - energies[0]

    return images_ase, final_fmax, energy_barrier


def run_neb_patience_energy(calc, n_images, fmax_thresh, patience, neb_steps):
    """
    NEB with early stopping:
      - Hard stop if fmax < fmax_thresh
      - Patience based on improvement of max energy along the band
    """
    # --- Build images ---
    images_ase = [copy.deepcopy(initial_ase)]
    for i in range(1, n_images - 1):
        image_ase = copy.deepcopy(initial_ase)
        if set_tags:
            image_ase.set_tags(np.ones(len(image_ase)))
        image_ase.set_calculator(copy.deepcopy(calc))
        images_ase.append(image_ase)
    final_copy = copy.deepcopy(final_ase)
    final_copy.set_calculator(copy.deepcopy(calc))
    images_ase.append(final_copy)

    # --- Set up NEB ---
    neb_ase = NEB(images_ase, climb=False, remove_rotation_and_translation=remove_rot, method=neb_method)
    neb_ase.interpolate(method='linear', apply_constraint = apply_constraint)
    qn_ase = FIRE(neb_ase)

    # --- Tracking ---
    energy_tol = 1e-8  # small tolerance to ignore numerical noise
    best_max_energy = np.inf
    no_improve = 0

    for step in range(neb_steps):
        qn_ase.step()

        # Forces / fmax
        forces = neb_ase.get_forces()
        fmax = norm(forces, axis=1).max()

        # Energies
        energies = [img.get_potential_energy() for img in images_ase]
        max_E = max(energies)
        min_E = min(energies)

        # Progress print
        print(f"Step {step:4d} | fmax = {fmax:.6f} | max(E) = {max_E:.6f} | min(E) = {min_E:.6f}")

        # Hard stop on fmax convergence
        if fmax < fmax_thresh:
            print("Stopping: reached fmax threshold")
            break

        # Patience on max energy improvement (we want max(E) to DECREASE)
        if max_E < best_max_energy - energy_tol:
            best_max_energy = max_E
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Stopping: no improvement in max(E) for {patience} steps")
            break

    # --- Final evaluation ---
    forces = neb_ase.get_forces()
    final_fmax = norm(forces, axis=1).max()
    energies = [img.get_potential_energy() for img in images_ase]
    energy_barrier = max(energies) - energies[0]

    return images_ase, final_fmax, energy_barrier


def run_neb_best(calc, n_images, neb_steps):
    """
    Run NEB for exactly `neb_steps` iterations, track the lowest fmax observed,
    and return that iteration's images, fmax, and energy barrier.

    Returns:
        images_ase_best : list[ase.Atoms]
        best_fmax       : float
        best_barrier    : float
    """

    # --- Build initial NEB images (same style as before) ---
    images_ase = [copy.deepcopy(initial_ase)]
    for i in range(1, n_images - 1):
        image_ase = copy.deepcopy(initial_ase)
        if set_tags:
            image_ase.set_tags(np.ones(len(image_ase)))
        image_ase.set_calculator(copy.deepcopy(calc))
        images_ase.append(image_ase)
    images_ase.append(copy.deepcopy(final_ase))

    neb_ase = NEB(
        images_ase,
        climb=False,
        remove_rotation_and_translation=remove_rot,
        method=neb_method,
    )
    neb_ase.interpolate(method="linear", apply_constraint = apply_constraint)
    qn_ase = FIRE(neb_ase)

    best_step = -1
    best_fmax = np.inf
    best_barrier = None
    best_positions = None

    for step in range(neb_steps):
        qn_ase.step()

        # --- Compute fmax using per-Atoms forces on the *mobile* images only ---
        """
        per_image_fmax = []
        for img in images_ase[1:-1]:
            F = img.get_forces()  # shape (n_atoms, 3)
            if F is None or F.size == 0:
                continue
            per_image_fmax.append(norm(F, axis=1).max())
        fmax = max(per_image_fmax) if per_image_fmax else 0.0
        """

        forces = neb_ase.get_forces()
        fmax = norm(forces, axis=1).max()

        # --- Energies & barrier at this step ---
        energies = [img.get_potential_energy() for img in images_ase]
        barrier = max(energies) - energies[0]

        # Print progress
        print(
            f"Step {step:4d} | fmax = {fmax:.6f} | barrier = {barrier:.6f} "
            f"| max(E) = {max(energies):.6f} | min(E) = {min(energies):.6f}"
        )

        # Track best snapshot
        if fmax < best_fmax:
            best_fmax = fmax
            best_step = step
            best_barrier = barrier
            # Store only positions to keep memory small
            best_positions = [img.get_positions().copy() for img in images_ase]


    # --- Rebuild images for the best iteration from saved positions ---
    if best_positions is None:
        print("No steps taken or unable to evaluate forces; returning current state.")
        energies = [img.get_potential_energy() for img in images_ase]
        energy_barrier = max(energies) - energies[0]
        # Recompute fmax from images (current state)
        per_image_fmax = []
        for img in images_ase[1:-1]:
            F = img.get_forces()
            if F is None or F.size == 0:
                continue
            per_image_fmax.append(norm(F, axis=1).max())
        final_fmax = max(per_image_fmax) if per_image_fmax else 0.0
        return images_ase, float(final_fmax), float(energy_barrier)

    images_ase_best = [copy.deepcopy(initial_ase)]
    for i in range(1, n_images - 1):
        img = copy.deepcopy(initial_ase)
        img.set_calculator(copy.deepcopy(ase_calculator)) #ase calc not calc since calc takes up too much space
        images_ase_best.append(img)
    images_ase_best.append(copy.deepcopy(final_ase))

    for img, pos in zip(images_ase_best, best_positions):
        img.set_positions(pos)

    print(
        f"\nBest fmax found at iteration {best_step} "
        f"(fmax={best_fmax:.6f}, barrier={best_barrier:.6f}). Returning that iteration."
    )

    return images_ase_best, float(best_fmax), float(best_barrier)

