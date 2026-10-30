#!/bin/bash
#SBATCH --<slurm details>

python bertGlue.py --task wnli --epochs 3 --batch 16
python gptGlue.py --task wnli --epochs 3 --batch 16
python t5Glue.py --task wnli --epochs 3 --batch 16

# example nonGLUE run
python gpt2_downstream.py \
    --task genia \
    --output outputs/pr_gpt2_genia_results.csv