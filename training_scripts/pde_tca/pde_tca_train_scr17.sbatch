#!/bin/bash
#SBATCH -c 4
#SBATCH --time=10:00:00
#SBATCH --job-name=pde_tca_run17
#SBATCH --output=outputs/pde_tca_run17_out.out
#SBATCH --mem=32G
#SBATCH --error=errs/pde_tca_run17_err.err

module load py-pytorch/2.0.0_py39

./training_scripts/pde_tca/pde_tca_train_scr17.sh