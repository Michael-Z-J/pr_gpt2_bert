# Punctuation restoration as a learning objective
# extended to models: T5, GPT2 and Bert

Testing effects of punctuation restoration on various downstream tasks.

## Tasks

The tasks that we cover are:
GLUE benchmark tasks: SST-2, QQP, MNLI (matched and mismatched), QNLI, RTE, WNLI, CoLA, MRPC and STS-B
non GLUE tasks:
Relation Extraction (RE)
Named Entity Recognition (NER)
Parts of Speech Tagging (POS)
Semantic Role Labeling (SRL)
Open Information Extraction (OIE)


## setup

python download.py downloads needed files for training

We train T5 through the code here: https://github.com/Aatlantise/pr-t5/tree/main
To train GPT2-Medium: python train_gpt2.py
To train Bert-base-uncased: python train_bert.py

The checkpoints created for the experiments should be:
./gpt2-medium
./pr-gpt2-output

./bert-base-uncased
./pr-bert-financial-news


## Experiments
Below are examples of running GLUE experiment wnli

python bertGlue.py --task wnli --epochs 3 --batch 16
python gptGlue.py --task wnli --epochs 3 --batch 16
python t5Glue.py --task wnli --epochs 3 --batch 16

To vary to other GLUE tasks, switch "wnli" for any of:
sst2, qqp, mnli_matched, mnli_mismatched, qnli, rte, wnli, cola, mrpc, stsb

Non-GLUE tasks can be run using the downstream files

E.g. for running ontonotes5:
#python bert_downstream.py --task ontonotes5 --output outputs/pr_bert_ontonotes5_results.csv
#python gpt2_downstream.py --task ontonotes5 --output outputs/pr_gpt2_ontonotes5_results.csv

The supported tasks are:
pos, ner, genia, ontonotes5


To vary models, pass in the flag --model followed by model name

To run OIE experiment, first clone the oie-benchmark before running oie_bert_BIO_tags.py or oie_gpt2.py

T5_train_eval.py is intended to be run using https://github.com/aatlantise/punc-rest-improves


