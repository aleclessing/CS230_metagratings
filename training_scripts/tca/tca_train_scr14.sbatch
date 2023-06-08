#!/bin/bash
#SBATCH -c 4
#SBATCH --time=10:00:00
#SBATCH --job-name=tca_run14
#SBATCH --output=outputs/tca_run14_out.out
#SBATCH --mem=32G
#SBATCH --error=errs/tca_run14_err.err

module load py-pytorch/2.0.0_py39

./training_scripts/tca/tca_train_scr14.sh