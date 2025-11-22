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

EXP_NAME="7_27_ELJ38_RandSamp100_NEB70_NumInterp20_LinearDev=0_03_Epochs=50"

singularity exec  --nv --bind /sdf/group/mli/pranav/ /sdf/group/mli/sgaz/images/fair-chem.sif python3 random_sampler.py --exp-name "$EXP_NAME"
