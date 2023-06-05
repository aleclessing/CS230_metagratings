#!/bin/bash
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --job-name=tca_run6
#SBATCH --output=outputs/tca_run6_out.out
#SBATCH --mem=32G
#SBATCH --error=errs/tca_run6_err.err

module load py-pytorch/2.0.0_py39

./training_scripts/tca/tca_train_scr6.sh