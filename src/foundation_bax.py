# NN-bax things
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
import random
from numpy.linalg import norm


# NEB things
from ase.io import read
from ase.neb import NEB

from ase.neb import NEBTools
import copy

from ase.calculators.lj import LennardJones

from ase import Atoms
from ase.optimize import LBFGS, MDMin, FIRE
from ase.io import Trajectory

from ase.optimize.basin import BasinHopping
from ase.optimize.minimahopping import MinimaHopping

from ase.visualize import view
from tqdm import tqdm
import sys
import os
import glob
import time
import subprocess
import sys

from fairchem.core import OCPCalculator
from fairchem.core.models.model_registry import available_pretrained_models

import pickle
from sklearn.decomposition import PCA
from scipy.interpolate import griddata

from fairchem.core import OCPCalculator
from fairchem.core.models.model_registry import available_pretrained_models
from fairchem.core.common.tutorial_utils import fairchem_main
from ase.io import read
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator

from pathlib import Path
from datetime import datetime
import yaml
import argparse


###--- STUFF TO CHANGE START -------
#exp_name = "7_23_SHJ7_BAX100_NEB180_LinearDev=0_03" #make command line argument 

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", required=True,
                    help="Experiment name string")
args = parser.parse_args()

exp_name = args.exp_name  # <-- use everywhere below exactly as before

#traj_dir = "/sdf/group/mli/pranav/local_minima_test.traj" #(0 to 5)
#traj_dir = "/sdf/group/mli/pranav/lj7_paths/one_min_thresh_fmax_0.08.traj" #(0 to 19)
#traj_dir = "LJ7_superhard_3rdIndex.traj" #(0 to 3)
#traj_dir = "5_23_LJ7_3minima_02.traj" #(0 to 2)
#traj_dir = "simple_LJ38.traj" #(1 to 0)
#traj_dir = "5_26_newEasyLJ38_fromGlobalmin_02_fmax=0.05.traj" #0 to 2
#traj_dir = "5_26_LJ38_oneLocalMin_throughGlobal_03_fmax=0.41.traj"#0 to 3
#traj_dir = "7_29_LJ38_oneShallowMin_20img_0to4_fmax=0p3.traj" #0 to 4, 60 NEB

#Parts 1 2 and 3 of two-min LJ38 trasition, all 0 to 1
#traj_dir = "8_7_10to12_FullLJ38_50NEB_fmax=0p25_20img_0to1.traj"
#traj_dir = "8_7_12to14_FullLJ38_40NEB_fmax=0p45_20img_0to1.traj"
#traj_dir = "8_7_14to16_FullLJ38_60NEB_fmax=0p05_20img_0to1.traj"

traj_dir_arr = ["8_7_10to12_FullLJ38_50NEB_fmax=0p25_20img_0to1.traj", "8_7_12to14_FullLJ38_40NEB_fmax=0p45_20img_0to1.traj", "8_7_14to16_FullLJ38_60NEB_fmax=0p05_20img_0to1.traj"]
n_paths = len(traj_dir_arr)

N_LJ = 38
str_lj = "Ar" + str(N_LJ)


BAX_iters = 70
fmax_thresh = 0.1
neb_steps = 80
do_lowest_fmax = True 

#t_mae = [0.1, 0.1, 0.1]
#t_fmax = [0.5, 0.90, 0.1]
#t_fmax = [0.5, 0.90, 0.3] #final one has to be bigger 
t_fmax = [0.7, 0.90, 0.5] #final one has to be bigger 
t_mae = [0.15, 0.15, 0.15]

n_images = [20, 20, 20]

initial_ase_0 = Trajectory(traj_dir_arr[0])[0]
initial_ase_1 = Trajectory(traj_dir_arr[1])[0]
initial_ase_2 = Trajectory(traj_dir_arr[2])[0]
final_ase_0 = Trajectory(traj_dir_arr[0])[1]
final_ase_1 = Trajectory(traj_dir_arr[1])[1]
final_ase_2 = Trajectory(traj_dir_arr[2])[1]


ase_calculator = LennardJones(rc=100)
initial_ase_0.set_calculator(copy.deepcopy(ase_calculator))
initial_ase_1.set_calculator(copy.deepcopy(ase_calculator))
initial_ase_2.set_calculator(copy.deepcopy(ase_calculator))
final_ase_0.set_calculator(copy.deepcopy(ase_calculator))
final_ase_1.set_calculator(copy.deepcopy(ase_calculator))
final_ase_2.set_calculator(copy.deepcopy(ase_calculator))

IS = [ initial_ase_0, initial_ase_1, initial_ase_2]
FS = [ final_ase_0, final_ase_1, final_ase_2]


###--- STUFF TO CHANGE END -------



ase_calculator = LennardJones(rc=100)

#Editing the Config of the Yaml File
input_file = "config_lj7_BAX.yml"
output_dir = "configs"

output_file = os.path.join(output_dir, f"config_{exp_name.strip('_')}.yml")

# Load the original YAML config
with open(input_file, "r") as f:
    config = yaml.safe_load(f)

# Recursive function to update all "train.db" entries
def replace_train_db(obj):
    if isinstance(obj, dict):
        return {k: replace_train_db(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_train_db(i) for i in obj]
    elif isinstance(obj, str) and obj == "train.db":
        return exp_name + "train.db"
    else:
        return obj

# Apply the change
modified_config = replace_train_db(config)

# Save the modified config
with open(output_file, "w") as f:
    yaml.dump(modified_config, f)

print(f"Saved modified config to {output_file}")



def run_neb(calc, n_images, fmax_thresh, neb_steps, initial_ase, final_ase):
    # 2.A. NEB using ASE
    n_images = n_images
    images_ase = [copy.deepcopy(initial_ase)]
    for i in range(1, n_images-1):
        image_ase = copy.deepcopy(initial_ase)
        image_ase.set_calculator(copy.deepcopy(calc))
        images_ase.append(image_ase)
    images_ase.append(copy.deepcopy(final_ase))

    neb_ase = NEB(images_ase, climb=False, remove_rotation_and_translation=True, method='string')
    neb_ase.interpolate(method='linear')
    qn_ase = FIRE(neb_ase)
    qn_ase.run(fmax=fmax_thresh, steps=neb_steps)

 
    
    # Extract final fmax from the NEB band
    forces = neb_ase.get_forces()
    final_fmax = norm(forces, axis=1).max()

    # Compute per‑image potentials and barrier
    energies = [img.get_potential_energy() for img in images_ase]
    energy_barrier = max(energies) - energies[0]

    return images_ase, final_fmax, energy_barrier


def run_neb_best(calc, n_images, neb_steps, initial_ase, final_ase):
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
        image_ase.set_calculator(copy.deepcopy(calc))
        images_ase.append(image_ase)
    images_ase.append(copy.deepcopy(final_ase))

    neb_ase = NEB(
        images_ase,
        climb=False,
        remove_rotation_and_translation=True,
        method="string",
    )
    neb_ase.interpolate(method="linear")
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




def pick_random_image(path, random_state): 
    # New: need random state to not just pick same im every time
    random.setstate(random_state)
    if not path:
        raise ValueError("Cannot pick from an empty path list.")

    out = random.choice(path)
    return out, random.getstate()

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


def train_fBAX(i, n, checkpoint_path):
    # New: output fn of i to avoid time grouping
    t0 = time.time()
    cmd = [
        sys.executable,
        "main.py",
        "--mode", "train",
        "--config-yml", output_file,
        "--checkpoint", checkpoint_path,
        "--run-dir", "/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{n}_{i}",
        "--identifier", "ft-lj7",
    ]

    print("IN TRAINING FILE, EXP NAME:",  exp_name )

    with open(f"train.txt", "w") as logf:
        result = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - t0
    print(f"Elapsed time = {elapsed:1.1f} seconds")

    if result.returncode != 0:
        print(f"\n⚠️  Training exited with code {result.returncode}. Last lines of log:")
        print("-" * 60)
        print(tail("train.txt", n=20))
        print("-" * 60)
        print("Please inspect the full `train.txt` for the full traceback.")


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
    sampled_image.set_calculator(lj_calc)
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


## ----- Actual Parameters and Code ---------


random_state = random.getstate()



#-------Pre-adding deviations from the linear interpolation-----

equiformer_calculator = OCPCalculator(
    checkpoint_path="/sdf/group/mli/pranav/pretrained_models/eqV2_153M_omat.pt",
    cpu=False,
) 
calculator = LennardJones(rc=100)

lj_calc = LennardJones(rc=100)     # one calculator instance
# build & write entries with energy, forces, and zero stress


fmax_history_arr = []
energy_barrier_history_arr = []
subset_history_arr = []
acquired_data_history_arr = []
mae_history_arr = []

#------- MeanBAX Loop for FoundationBAX-------
for n in range(n_paths):

    #Set the starting model for foundationBAX 
    if n == 0:
        foundation_checkpoint = "/sdf/group/mli/pranav/pretrained_models/eqV2_153M_omat.pt"
    else:
        foundation_checkpoint = latest_checkpoint_path

    acquired_data = [add_zero_stress_spc(copy.deepcopy(IS[n])), add_zero_stress_spc(copy.deepcopy(FS[n]))]
    train_db_path = exp_name + "train.db"
    db = connect(train_db_path, append=False)
    db.write(add_zero_stress_spc(copy.deepcopy(IS[n])))
    db.write(add_zero_stress_spc(copy.deepcopy(FS[n])))


    fmax_history = []
    energy_barrier_history = []
    subset_history = []
    acquired_data_history = []
    mae_history = []


    for i in range(BAX_iters):   

        print("Running BAX Iter: ", i, "for Path ", n)

        #3: Write data to db
        db = connect(train_db_path, append=False)
        for struct in acquired_data:
            db.write(struct)

        #4. Train Equiformer on train.db
        train_fBAX(i, n, foundation_checkpoint)
        
        if i != 0:
            old_checkpoint = latest_checkpoint
            old_checkpoint_path = latest_checkpoint_path
            equiformer_calculator_n_minus1 = OCPCalculator(checkpoint_path=old_checkpoint_path, 
                        seed=42, 
                        cpu=False, 
                        trainer='ocp')
        else:
            equiformer_calculator_n_minus1 = None

        latest_checkpoint = find_latest_best_ckpt("/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{n}_{i}/checkpoints/")
        latest_checkpoint_path = "/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{n}_{i}/checkpoints/" + latest_checkpoint + "/best_checkpoint.pt"
                
        print("Latest Checkpoint: ", latest_checkpoint_path)
    
        equiformer_calculator_n = OCPCalculator(checkpoint_path=latest_checkpoint_path, 
                        seed=42, 
                        cpu=False, 
                        trainer='ocp')
        

        
        
        #Evaluate 

        #5. Delete old model
        if i != 0:
            delete_checkpoints_in_dir("/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{i-1}/checkpoints/" + old_checkpoint)
            print("Deleted Old Model!")
    

        #1. RUN NEB on Trained Model
        #path = run_neb(equiformer_calculator_n, n_images, fmax_thresh, neb_steps)

        if do_lowest_fmax:
            path, final_fmax, energy_barrier = run_neb_best(equiformer_calculator_n, n_images[n], neb_steps, IS[n], FS[n])
        else:
            path, final_fmax, energy_barrier = run_neb(equiformer_calculator_n, n_images[n], fmax_thresh, neb_steps, IS[n], FS[n])

        fmax_history.append(final_fmax)
        energy_barrier_history.append(energy_barrier)

        print("MEMORY CHECK 1:", torch.cuda.memory_allocated()/(1024*1024*1024))  

        #2. Randomly Sample Image from Path
        sampled_image, random_state = pick_random_image(path[1:-1], random_state) #Don't pick the start or end
        sampled_image.set_calculator(lj_calc) #Didn't have before 7/26, actually samples LJ point
        sampled_image = add_zero_stress_spc(sampled_image) #Add stress for equiformer training
    

        acquired_data.append(sampled_image)
        
        
        #2.5: Save Stuff
        positions_array = np.array([atoms.get_positions() for atoms in path])
        subset_history.append(positions_array)
        acquired_data_history.append(sampled_image.get_positions())


        if equiformer_calculator_n_minus1 is not None:
            sample_copy = sampled_image.copy()
            mae_Nm1_on_N = compute_sample_mae(sample_copy, equiformer_calculator_n_minus1)
            mae_history.append(mae_Nm1_on_N)

        

        del(path) #get rid of path in memory
        print("MEMORY CHECK 2:", torch.cuda.memory_allocated()/(1024*1024*1024))  


        #CHECK CONVERGENCE CRITERION, IF SATISFIED THEN BREAK
        if i > 0 and mae_Nm1_on_N < t_mae[n] and final_fmax < t_fmax[n]:
            break


    fmax_history_arr.append(fmax_history)
    energy_barrier_history_arr.append(energy_barrier_history)
    subset_history_arr.append(subset_history)
    acquired_data_history_arr.append(acquired_data)
    mae_history_arr.append(mae_history)

    with open("results/" + exp_name + "_subset_history.pkl", "wb") as f:
        pickle.dump(subset_history_arr, f)
        print("Subset history saved in " + exp_name + "_subset_history.pkl")
        
    with open("results/" + exp_name +  "_acquired_data.pkl", "wb") as f:
        pickle.dump(acquired_data_history_arr, f)
        print("Acquired Data Saved in " + exp_name +  "_acquired_data.pkl")

    with open("results/" + exp_name + "_fmax_history.pkl", "wb") as f:
        pickle.dump(fmax_history_arr, f)
        print("Fmax history saved in " + exp_name + "_fmax_history.pkl")
        
    with open("results/" + exp_name +  "_energy_barrier_history_data.pkl", "wb") as f:
        pickle.dump(energy_barrier_history_arr, f)
        print("Energy Barrier Data Saved in " + exp_name +  "_energy_barrier_history_data.pkl")

    with open("results/" + exp_name +  "_mae_Nm1_on_N.pkl", "wb") as f:
        pickle.dump(mae_history_arr, f)
        print("Modeling Error Saved in " + exp_name +  "_mae_Nm1_on_N.pkl")
        
        
        
    
        
        

    
        


