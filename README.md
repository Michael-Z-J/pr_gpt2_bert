Testing effects of punctuation restoration on various downstream tasks.

The tasks that we cover are:
GLUE benchmark tasks: SST-2, QQP, MNLI (matched and mismatched), QNLI, RTE, WNLI, CoLA, MRPC and STS-B
non GLUE tasks:
Relation Extraction (RE)
Named Entity Recognition (NER)
Parts of Speech Tagging (POS)
Semantic Role Labeling (SRL)
Open Information Extraction (OIE)

python download.py downloads needed files for training

We train T5 through the code here: https://github.com/Aatlantise/pr-t5/tree/main
To train GPT2-Medium: python train_gpt2.py
To train Bert-base-uncased: python train_bert.py

See examples for running other tasks in run.sh
