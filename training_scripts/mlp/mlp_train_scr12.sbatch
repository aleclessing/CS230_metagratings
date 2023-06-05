#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1

#SBATCH --time=8:00:00
#SBATCH --job-name=mlp_run12
#SBATCH --output=outputs/mlp_run12_out.out
#SBATCH --mem=16G
#SBATCH --error=errs/mlp_run12_err.err

module load py-pytorch/2.0.0_py39

./training_scripts/mlp/mlp_train_scr12.sh