#!/bin/bash
#SBATCH -c 4
#SBATCH --time=8:00:00
#SBATCH --job-name=mlp_run32
#SBATCH --output=outputs/mlp_run32_out.out
#SBATCH --mem=16G
#SBATCH --error=errs/mlp_run32_err.err

module load py-pytorch/2.0.0_py39

./training_scripts/mlp/mlp_train_scr32.sh