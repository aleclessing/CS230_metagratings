#!/bin/bash
#
#SBATCH --job-name=training_test
#
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

srun echo "Executing train.py..."
srun python3 train.py
srun echo "Training complete!"
