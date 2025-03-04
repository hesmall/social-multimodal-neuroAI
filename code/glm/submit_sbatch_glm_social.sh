#!/bin/bash

sbatch --array=1-$(( $( wc -l /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/data/participants.tsv | cut -f1 -d' ' ) - 1 )) sbatch_glm_social.sh Sherlock_ASD