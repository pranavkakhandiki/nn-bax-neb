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

EXP_NAME="9_4_LJ38TwoMin_BAX70_MaxNEB80_fmax=0p7_0p9_0p5_tmae=0p15_FoundationBAX"

singularity exec  --nv --bind /sdf/group/mli/pranav/ /sdf/group/mli/sgaz/images/fair-chem.sif python3 foundation_bax.py --exp-name "$EXP_NAME"
