#!/bin/bash
STUDY=$1 #first input is study folder

sbatch --array=1-$(( $( wc -l /home/hsmall2/deep_nat_lat/data/participants.tsv | cut -f1 -d' ' ) - 1 )) sbatch_encoding_pipeline1.sh Sherlock_ASD