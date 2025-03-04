#!/bin/bash
#SBATCH --mail-type=END 			# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hsmall2@jhu.edu	# Your email address
#SBATCH --nodes=1				# OpenMP requires a single node
#SBATCH --ntasks-per-node=1
#SBATCH --partition parallel
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=180G
#SBATCH --time=04:00:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=ISC			# Descriptive job name
##### END OF JOB DEFINITION  #####
POP=$1 #first input is study folder
SMOOTH=$2

module --ignore_cache load "anaconda"
# ml anaconda
conda activate sherlock_env

python -u intersubject_correlation.py --population $POP \
		--dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat \
		--out_dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/analysis \
		--figure_dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/figures \
		--smoothing-fwhm $SMOOTH \
		--mask None

conda deactivate