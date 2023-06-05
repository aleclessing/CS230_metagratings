#!/bin/bash
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --job-name=tca_run9
#SBATCH --output=outputs/tca_run9_out.out
#SBATCH --mem=16G
#SBATCH --error=errs/tca_run9_err.err

module load py-pytorch/2.0.0_py39

./training_scripts/tca/tca_train_scr9.sh