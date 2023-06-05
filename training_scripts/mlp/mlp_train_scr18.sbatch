#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1

#SBATCH --time=8:00:00
#SBATCH --job-name=mlp_run18
#SBATCH --output=outputs/mlp_run18_out.out
#SBATCH --mem=16G
#SBATCH --error=errs/mlp_run18_err.err

module load py-pytorch/2.0.0_py39

./training_scripts/mlp/mlp_train_scr18.sh