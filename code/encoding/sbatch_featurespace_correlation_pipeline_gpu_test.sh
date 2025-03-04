#!/bin/bash
#SBATCH --mail-type=END 			# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hsmall2@jhu.edu	# Your email address
#SBATCH --nodes=1					# OpenMP requires a single node
#SBATCH --ntasks=1					# Run a single serial task
#SBATCH --partition=a100
#SBATCH -A lisik3_gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1:00:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=feat_reg			# Descriptive job name
##### END OF JOB DEFINITION  #####
# STUDY=$1 #first input is study folder
# MODEL=$2 #second input is the model to run

module --ignore_cache load "anaconda"
# ml anaconda
conda activate sherlock_env

features=$( sed -n "$((${SLURM_ARRAY_TASK_ID} + 1))p" /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/code/encoding/featurespace_comparisons_both_ways_test.tsv)

python -u featurespace_correlation.py \
		--features $features \
		--dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat \
		--out_dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/analysis \
		--figure_dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/figures \
		--chunklen 20 \
		--method b2b_regression

conda deactivate