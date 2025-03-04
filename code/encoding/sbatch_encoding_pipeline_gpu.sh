#!/bin/bash
#SBATCH --mail-type=END 			# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hsmall2@jhu.edu	# Your email address
#SBATCH --nodes=1					# OpenMP requires a single node
#SBATCH --ntasks=1					# Run a single serial task
#SBATCH --partition=a100
#SBATCH -A lisik3_gpu
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=160G
#SBATCH --time=18:00:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=encoding			# Descriptive job name
##### END OF JOB DEFINITION  #####
STUDY=$1 #first input is study folder
MODEL=$2 #second input is the model to run

module --ignore_cache load "anaconda"
# ml anaconda
conda activate sherlock_env

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/data/participants.tsv )

echo "Job running on node: $(hostname)" >> sbatch_logs/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

python -u encoding.py --s_num $subject \
		--dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat \
		--data_dir /home/hsmall2/scratch4-lisik3/$STUDY/data \
		--out_dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/analysis \
		--figure_dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/figures \
		--model $MODEL \
		--smoothing-fwhm 3.0 \
		--mask ISC

conda deactivate