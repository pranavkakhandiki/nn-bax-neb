#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:bax
#SBATCH --job-name=neb
#SBATCH --output=outputs/output-%j.txt
#SBATCH --error=outputs/output-%j.txt
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=150:00:00



# Ensure the outputs directory exists
mkdir -p outputs

EXP_NAME="10_30_ELJ38_random_delta=0p2"
#EXP_NAME="10_14_EAM_Cu38_dimerNew"

singularity exec  --nv --bind /sdf/group/mli/pranav/ /sdf/group/mli/sgaz/images/fair-chem-v1.10.0.sif python3 nn_bax.py --exp-name "$EXP_NAME"
