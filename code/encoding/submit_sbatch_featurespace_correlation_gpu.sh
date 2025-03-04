#!/bin/bash
# MODEL=$1 #first input is model
#array=1-X%8 only lets 8 run at once -- saving gpus for other people
sbatch --array=1-$(( $( wc -l /home/hsmall2/scratch4-lisik3/hsmall2/deep_nat_lat/code/encoding/featurespace_comparisons_both_ways.tsv | cut -f1 -d' ' ) - 1 )) sbatch_featurespace_correlation_pipeline_gpu.sh 