#!/bin/bash
#SBATCH --mem 32GB        
#SBATCH --time=0-48:00   
#SBATCH --job-name=log_{job_id}
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH -p gpu-small
#SBATCH --wckey=quams
#SBATCH -c 1
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate $MENV     # This step loads the Python environment.
echo "$@"
python perform_stability_tests.py "$@"
