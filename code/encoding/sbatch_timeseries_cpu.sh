#!/bin/bash
#SBATCH --mail-type=END 			# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hsmall2@jhu.edu	# Your email address
#SBATCH --nodes=1				# OpenMP requires a single node
#SBATCH --ntasks-per-node=1
#SBATCH --partition shared
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00				# Time limit hh:mm:ss
#SBATCH -e sbatch_logs/error_%A_%a.log			# Standard error
#SBATCH -o sbatch_logs/output_%A_%a.log			# Standard output
#SBATCH --job-name=time_series			# Descriptive job name
##### END OF JOB DEFINITION  #####
STUDY=$1 #first input is study folder

module --ignore_cache load "anaconda"
# ml anaconda
conda activate sherlock_env

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/data/participants.tsv )

python -u timeseries.py --s_num $subject \
		--dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat \
		--data_dir /home/hsmall2/scratch4-lisik3/$STUDY/data \
		--out_dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/analysis \
		--figure_dir /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/figures \
		--smoothing-fwhm 6.0 \
		--mask None

conda deactivate