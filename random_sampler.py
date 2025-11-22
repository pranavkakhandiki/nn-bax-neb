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
traj_dir = "simple_LJ38.traj" #(1 to 0)
#traj_dir = "5_26_newEasyLJ38_fromGlobalmin_02_fmax=0.05.traj" #0 to 2
#traj_dir = "5_26_LJ38_oneLocalMin_throughGlobal_03_fmax=0.41.traj"#0 to 3

#RANDOM SAMPLER FILE
N_LJ = 38
str_lj = "Ar" + str(N_LJ)
initial_traj_num = 1
final_traj_num = 0

BAX_iters = 100
neb_steps = 70
n_images = 20

n_interp = n_images #Number of images to start with 
delta_var = 0.03 #Amount to vary starting points

fmax_thresh = 0.15

neb_steps_array = np.concatenate([
    np.linspace(1, 20, num=20, dtype=int),  # First 20 steps linearly increasing
    np.full(BAX_iters - 20, 20, dtype=int)  # Remaining steps fixed at 20
])


###--- STUFF TO CHANGE END -------



ase_calculator = LennardJones(rc=100)
test = Trajectory(traj_dir)

initial_ase = copy.deepcopy(test[initial_traj_num])
initial_ase.set_calculator(copy.deepcopy(ase_calculator))
final_ase = copy.deepcopy(test[final_traj_num])
final_ase.set_calculator(copy.deepcopy(ase_calculator))

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



def run_neb(calc, n_images, fmax_thresh, neb_steps):
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
    
    return images_ase



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

def train(i):
    # New: output fn of i to avoid time grouping
    t0 = time.time()
    cmd = [
        sys.executable,
        "main.py",
        "--mode", "train",
        "--config-yml", output_file,
        "--checkpoint", "/sdf/group/mli/pranav/pretrained_models/eqV2_153M_omat.pt",
        "--run-dir", "/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{i}",
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


## ----- Actual Parameters and Code ---------


random_state = random.getstate()

train_db_path = exp_name + "train.db"
db = connect(train_db_path, append=False)

db.write(add_zero_stress_spc(copy.deepcopy(initial_ase)))
db.write(add_zero_stress_spc(copy.deepcopy(final_ase)))

subset_history = []
acquired_data_history = []
acquired_data = [add_zero_stress_spc(copy.deepcopy(initial_ase)), add_zero_stress_spc(copy.deepcopy(final_ase))]

#-------Pre-adding deviations from the linear interpolation-----
print("Adding augmentations from linear data")

def perturb_positions(positions, delta):

    perturbation = np.random.uniform(-delta, delta, positions.shape)  # Generate perturbations
    return positions + perturbation


alphas = np.linspace(0, 1, n_interp)
linear_images_ase = []
for alpha in alphas:
    # Interpolate positions between initial and final
    pos_interp = (1 - alpha) * initial_ase.positions + alpha * final_ase.positions
    # Create a new Atoms object from the initial one and update positions
    atoms_interp = initial_ase.copy()
    atoms_interp.positions = pos_interp
    linear_images_ase.append(atoms_interp)

equiformer_calculator = OCPCalculator(
    checkpoint_path="/sdf/group/mli/pranav/pretrained_models/eqV2_153M_omat.pt",
    cpu=False,
) 
calculator = LennardJones(rc=100)
lj_calc = LennardJones(rc=100)     # one calculator instance

#------- MeanBAX Loop -------
for i in range(BAX_iters):   

    print("Running BAX Iter: ", i)

    #3: Write data to db
    db = connect(train_db_path, append=False)
    for struct in acquired_data:
        db.write(struct)

    #4. Train Equiformer on train.db
    train(i)
    
    if i != 0:
        old_checkpoint = latest_checkpoint

    latest_checkpoint = find_latest_best_ckpt("/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{i}/checkpoints/")
    latest_checkpoint_path = "/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{i}/checkpoints/" + latest_checkpoint + "/best_checkpoint.pt"
            
    print("Latest Checkpoint: ", latest_checkpoint_path)
  
    equiformer_calculator_n = OCPCalculator(checkpoint_path=latest_checkpoint_path, 
                     seed=42, 
                     cpu=False, 
                     trainer='ocp')

    #5. Delete old model
    if i != 0:
        delete_checkpoints_in_dir("/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{i-1}/checkpoints/" + old_checkpoint)
        print("Deleted Old Model!")


    #1. RUN NEB on Trained Model
    path = run_neb(equiformer_calculator_n, n_images, fmax_thresh, neb_steps)
    print("MEMORY CHECK 1:", torch.cuda.memory_allocated()/(1024*1024*1024))  

    #2. Randomly Sample Image from Path
    #sampled_image, random_state = pick_random_image(path[1:-1], random_state) #Don't pick the start or end
    #sampled_image.set_calculator(lj_calc) #Didn't have before 7/26, actually samples LJ point
    #sampled_image = add_zero_stress_spc(sampled_image) #Add stress for equiformer training
   

    path_index = i % n_images    
    pos = perturb_positions(linear_images_ase[path_index].positions, delta_var)
    sampled_image = Atoms(str_lj, positions=pos)
    sampled_image.set_calculator(lj_calc)
    sampled_image = add_zero_stress_spc(sampled_image)

    acquired_data.append(sampled_image)
    
    
    #2.5: Save Stuff
    positions_array = np.array([atoms.get_positions() for atoms in path])
    subset_history.append(positions_array)
    acquired_data_history.append(sampled_image.get_positions())

    del(path) #get rid of path in memory
    print("MEMORY CHECK 2:", torch.cuda.memory_allocated()/(1024*1024*1024))  

    with open("results/" + exp_name + "_subset_history.pkl", "wb") as f:
        pickle.dump(subset_history, f)
        print("Subset history saved in " + exp_name + "_subset_history.pkl")
        
    with open("results/" + exp_name +  "_acquired_data.pkl", "wb") as f:
        pickle.dump(acquired_data_history, f)
        print("Acquired Data Saved in " + exp_name +  "_acquired_data.pkl")
    
    
   
    
    

   
    


