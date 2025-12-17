# Efficient Nudged Elastic Band Method using Neural Network Bayesian Algorithm Execution
Nudged Elastic Band (NEB) with Neural Network Bayesian Algorithm Execution (NN-BAX)

This repository contains an implementation of **Neural Network Bayesian Algorithm Execution (NN-BAX)** for accelerating **nudged elastic band (NEB)** calculations in atomistic systems. NN-BAX is an algorithm-aware active learning framework that combines Bayesian Algorithm Execution with large, symmetry-aware neural network force fields to efficiently discover minimum energy pathways (MEPs) between metastable states. By actively selecting training points that directly improve the NEB solution, NN-BAX significantly reduces the number of expensive energy and force evaluations required by classical NEB while preserving the accuracy of the resulting transition pathways.

The code in this repository demonstrates NN-BAX on both **Lennard-Jones (LJ)** cluster transitions and **Embedded Atom Method (EAM)** surface diffusion transitions. These systems span a range of dimensionalities and physical complexity, from well-studied LJ test cases to many-body metallic diffusion processes, and serve as controlled benchmarks for evaluating the efficiency and robustness of the method.

## Running NN-BAX

The main entry point for running NN-BAX is the file `nn_bax.py`. This script executes the full NN-BAX loop, including initialization of a pretrained foundation model, running NEB using the neural network surrogate, acquiring targeted samples from the predicted pathway using BAX, and iteratively fine-tuning the model until convergence criteria are met.

In typical usage, `nn_bax.py` is executed as a batch job on an HPC system using SLURM. The provided script `neb_job.sh` can be used to submit NN-BAX runs via `sbatch` and handles resource allocation and job configuration.

## Analysis and Example Notebooks

This repository includes example analysis workflows for evaluating NN-BAX results. The notebook `9-1 ELJ38 Best fmax method 200 Journal Suite.ipynb` provides a representative post-processing pipeline for analyzing the quality of NN-BAX-predicted pathways. The notebook analyzes an LJ38 transition using the trajectory stored in `simple_LJ38.traj`, and demonstrates how to inspect energy profiles along the path and compare NN-BAX results to classical NEB.

## Foundation-BAX for Multi-Step Transitions

For complex transitions involving multiple intermediate minima, this repository includes an implementation of **Foundation-BAX** in `foundation_bax.py`. Foundation-BAX extends NN-BAX to multi-step pathways by reusing the fine-tuned model from one sub-transition as the initialization for the next. This enables information learned from earlier transitions to be transferred across related pathways, further reducing the total number of required simulations and improving convergence for segmented or long transition paths.

