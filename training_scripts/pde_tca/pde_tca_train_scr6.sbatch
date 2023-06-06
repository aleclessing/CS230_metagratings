#!/bin/bash
#SBATCH -c 4
#SBATCH --time=10:00:00
#SBATCH --job-name=pde_tca_run6
#SBATCH --output=outputs/pde_tca_run6_out.out
#SBATCH --mem=32G
#SBATCH --error=errs/pde_tca_run6_err.err

module load py-pytorch/2.0.0_py39

./training_scripts/pde_tca/pde_tca_train_scr6.sh