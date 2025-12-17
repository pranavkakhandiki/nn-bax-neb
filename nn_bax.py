# NN-bax things
import sys

import torch
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

from neb import run_neb, run_neb_patience, run_neb_patience_energy, run_neb_best
from utils import random_sample_linear, pick_random_image, add_zero_stress_spc, tail, find_latest_best_ckpt, delete_checkpoints_in_dir, compute_sample_mae

from ase.visualize import view
from tqdm import tqdm
import sys
import os
import glob
import time
from time import perf_counter
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
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import FIRE
from ase.calculators.eam import EAM


from pathlib import Path
from datetime import datetime
import yaml
import argparse
import gc


###--- STUFF TO CHANGE START -------
#exp_name = "7_23_SHJ7_BAX100_NEB180_LinearDev=0_03" #make command line argument 

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", required=True,
                    help="Experiment name string")
args = parser.parse_args()

exp_name = args.exp_name  # <-- use everywhere below exactly as before

with open("nn_bax_config.yml", "r") as f:
    cfg = yaml.safe_load(f)


equi_checkpoint = cfg["equi_checkpoint"]
input_file = cfg["input_file"]
do_lowest_fmax = cfg["do_lowest_fmax"]
BAX_iters = cfg["BAX_iters"]
random_sample_method = cfg["random_sample_method"]
delta_var = cfg["delta_var"]


traj_dir = cfg["traj"]["traj_dir"]
N_LJ = cfg["traj"]["N_LJ"]
neb_system = cfg["traj"]["NEB_system"]
str_lj = "Ar" + str(N_LJ)
initial_traj_num = cfg["traj"]["initial_traj_num"]
final_traj_num = cfg["traj"]["final_traj_num"]
n_images = cfg["traj"]["n_images"]
fmax_thresh = cfg["traj"]["fmax_thresh"]
neb_steps = cfg["traj"]["neb_steps"]



#Not using anymore------

#Patience 
n_interp = n_images #Number of images to start with 
n_augmentations = 0 #augmentations around linear path to generate, set to zero for no initial sampling

###--- STUFF TO CHANGE END -------
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

#Editing the Config of the Yaml File
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


ELJ7_checkpoint = "/sdf/group/mli/pranav/7_26_ELJ7_BAX80_NEB25_NumInterp0_LinearDev=0_Epochs=50_fixedSample/lj7-finetune79/checkpoints/2025-07-27-14-32-48-ft-lj7/best_checkpoint.pt"
HLJ7_checkpoint = "/sdf/group/mli/pranav/7_26_HLJ7_BAX80_NEB55_NumInterp0_LinearDev=0_Epochs=50_fixedSample/lj7-finetune79/checkpoints/2025-07-27-19-05-52-ft-lj7/best_checkpoint.pt"


def train(i):
    # New: output fn of i to avoid time grouping
    t0 = time.time()
    cmd = [
        sys.executable,
        "main.py",
        "--mode", "train",
        "--config-yml", output_file,
        "--checkpoint", equi_checkpoint, #commented out for from-scratch-test
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
        print(tail("train.txt", n=200))
        print("-" * 60)
        print("Please inspect the full `train.txt` for the full traceback.")


## ----- Actual Parameters and Code ---------


random_state = random.getstate()

train_db_path = exp_name + "train.db"
db = connect(train_db_path, append=False)

db.write(add_zero_stress_spc(copy.deepcopy(initial_ase)))
db.write(add_zero_stress_spc(copy.deepcopy(final_ase)))

fmax_history = []
energy_barrier_history = []
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

calculator = LennardJones(rc=100)
lj_calc = LennardJones(rc=100)     # one calculator instance
# build & write entries with energy, forces, and zero stress

mae_history = []
training_time_history = []   # seconds spent inside `train(i)` for each iteration
loop_time_history     = []

#------- MeanBAX Loop -------
for i in range(BAX_iters):   

    print("Running BAX Iter: ", i)
    loop_t0 = perf_counter()

    #3: Write data to db
    db = connect(train_db_path, append=False)
    for struct in acquired_data:
        db.write(struct)

    #4. Train Equiformer on train.db
    train_t0 = perf_counter()
    train(i)
    training_time_history.append(perf_counter() - train_t0)
    
    if i != 0:
        old_checkpoint = latest_checkpoint
        old_checkpoint_path = latest_checkpoint_path
        equiformer_calculator_n_minus1 = OCPCalculator(checkpoint_path=old_checkpoint_path, 
                     #seed=42, 
                     cpu=False, 
                     trainer='ocp')
    else:
        equiformer_calculator_n_minus1 = None

    latest_checkpoint = find_latest_best_ckpt("/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{i}/checkpoints/")
    latest_checkpoint_path = "/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{i}/checkpoints/" + latest_checkpoint + "/best_checkpoint.pt"
            
    print("Latest Checkpoint: ", latest_checkpoint_path)
  
    equiformer_calculator_n = OCPCalculator(checkpoint_path=latest_checkpoint_path, 
                     #seed=42, 
                     cpu=False, 
                     trainer='ocp')
    
    #Evaluate 

    #5. Delete old model (not doing for final runs)
    if i != 0:
        delete_checkpoints_in_dir("/sdf/group/mli/pranav/" + exp_name + f"/lj7-finetune{i-1}/checkpoints/" + old_checkpoint)
        print("Deleted Old Model!")
 

    #1. RUN NEB on Trained Model
    #path = run_neb(equiformer_calculator_n, n_images, fmax_thresh, neb_steps)

    if do_lowest_fmax:
        path, final_fmax, energy_barrier = run_neb_best(equiformer_calculator_n, n_images, neb_steps)
    else:
        path, final_fmax, energy_barrier = run_neb(equiformer_calculator_n, n_images, fmax_thresh, neb_steps)

    fmax_history.append(final_fmax)
    energy_barrier_history.append(energy_barrier)

    print("MEMORY CHECK 1:", torch.cuda.memory_allocated()/(1024*1024*1024))  

    #2. Randomly Sample Image from Path
    if random_sample_method:
        sampled_image = random_sample_linear(linear_images_ase, delta_var)
    else:
        sampled_image, random_state = pick_random_image(path[1:-1], random_state) #Don't pick the start or end
        sampled_image.set_calculator(ase_calculator) #Didn't have before 7/26, actually samples LJ point
        sampled_image = add_zero_stress_spc(sampled_image) #Add stress for equiformer training
    

    acquired_data.append(sampled_image)
    
    
    #2.5: Save Stuff
    positions_array = np.array([atoms.get_positions() for atoms in path])
    subset_history.append(positions_array)
    acquired_data_history.append(sampled_image.get_positions())


    if equiformer_calculator_n_minus1 is not None:
        sample_copy = sampled_image.copy()
        with torch.no_grad():
            mae_Nm1_on_N = compute_sample_mae(sample_copy, equiformer_calculator_n_minus1)
        mae_history.append(mae_Nm1_on_N)
        del sample_copy, equiformer_calculator_n_minus1                           # NEW

    for img in path:                                                             # NEW
        img.set_calculator(None)                                                 
    del path 

    del equiformer_calculator_n                                                  # NEW

    print("MEMORY CHECK 2:", torch.cuda.memory_allocated()/(1024*1024*1024))  
    this_loop_secs = perf_counter() - loop_t0
    loop_time_history.append(this_loop_secs)

    with open("results/" + exp_name + "_subset_history.pkl", "wb") as f:
        pickle.dump(subset_history, f)
        print("Subset history saved in " + exp_name + "_subset_history.pkl")
        
    with open("results/" + exp_name +  "_acquired_data.pkl", "wb") as f:
        pickle.dump(acquired_data_history, f)
        print("Acquired Data Saved in " + exp_name +  "_acquired_data.pkl")

    with open("results/" + exp_name + "_fmax_history.pkl", "wb") as f:
        pickle.dump(fmax_history, f)
        print("Fmax history saved in " + exp_name + "_fmax_history.pkl")
        
    with open("results/" + exp_name +  "_energy_barrier_history_data.pkl", "wb") as f:
        pickle.dump(energy_barrier_history, f)
        print("Energy Barrier Data Saved in " + exp_name +  "_energy_barrier_history_data.pkl")

    with open("results/" + exp_name +  "_mae_Nm1_on_N.pkl", "wb") as f:
        pickle.dump(mae_history, f)
        print("Modeling Error Saved in " + exp_name +  "_mae_Nm1_on_N.pkl")

    with open("results/" + exp_name + "_training_time_history.pkl", "wb") as f:
        pickle.dump(training_time_history, f)

    with open("results/" + exp_name + "_loop_time_history.pkl", "wb") as f:
        pickle.dump(loop_time_history, f)

    
    
    
   
    
    

   
    


