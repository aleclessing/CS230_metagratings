#!/bin/bash
#SBATCH -c 4
#SBATCH --time=16:00:00
#SBATCH --job-name=pde_mlp_run4
#SBATCH --output=outputs/pde_mlp_run4_out.out
#SBATCH --mem=64G
#SBATCH --error=errs/pde_mlp_run4_err.err

module load py-pytorch/2.0.0_py39

./training_scripts/pde_mlp/pde_mlp_train_scr4.sh