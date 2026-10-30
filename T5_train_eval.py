import json
import os
import random
import re

from argparse import ArgumentParser
from importlib import import_module

import lightning
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from data.modules import TrainData, NumericTrainData, IntTrainData
from models.t5 import PRT5, PRT5Numeric
from models.felflarebert import felflarebert
from tasks.ner import score as object_generation_score
from utils import logger, clean_split


logger = logger()
torch.autograd.set_detect_anomaly(True)


DEFAULT_DATA_PATHS = {
    'pr': 'outputs/datasets/wiki-20231101.en-pr.jsonl',
    'mlm': 'outputs/datasets/wiki-20231101.en-mlm.jsonl',
    'srl': 'outputs/datasets/conll-2012-srl.jsonl',
    'pos': 'outputs/datasets/conll-2003-pos.jsonl',
    'oie': 'outputs/datasets/oie-2016-oie.jsonl',
    'chunking': 'outputs/datasets/conll-2000-chunking.jsonl',
    're': 'outputs/datasets/conll-2004-re.jsonl',
    'ner': 'outputs/datasets/conll-2003-ner.jsonl',
    'glue_CoLA': 'outputs/datasets/cola_train.jsonl',
    'glue_sst2': 'outputs/datasets/sst2_train.jsonl',
    'glue_mrpc': 'outputs/datasets/mrpc_train.jsonl',
    'glue_stsb': 'outputs/datasets/stsb_train.jsonl',
    'glue_qqp': 'outputs/datasets/qqp_train.jsonl',
    'glue_mnli': 'outputs/datasets/mnli_train.jsonl',
    'glue_qnli': 'outputs/datasets/qnli_train.jsonl',
    'glue_rte': 'outputs/datasets/rte_train.jsonl',
    'glue_wnli': 'outputs/datasets/wnli_train.jsonl',
}

GLUE_CLASSIFICATION_TASKS = {
    'glue_CoLA',
    'glue_sst2',
    'glue_mrpc',
    'glue_qqp',
    'glue_mnli',
    'glue_qnli',
    'glue_rte',
    'glue_wnli',
}

GLUE_DATA_PATHS = {
    'outputs/datasets/cola_train.jsonl',
    'outputs/datasets/sst2_train.jsonl',
    'outputs/datasets/mrpc_train.jsonl',
    'outputs/datasets/stsb_train.jsonl',
    'outputs/datasets/qqp_train.jsonl',
    'outputs/datasets/mnli_train.jsonl',
    'outputs/datasets/qnli_train.jsonl',
    'outputs/datasets/rte_train.jsonl',
    'outputs/datasets/wnli_train.jsonl',
}


def set_seed(seed: int):
    """Seed Python, NumPy, PyTorch, CUDA, Lightning, and DataLoader workers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    lightning.seed_everything(seed, workers=True)


def apply_prefix(task, x):
    return {'source': f"{task}: {x['source']}", 'target': x['target']}


def multitask_score(texts, outputs, targets, printer=print):
    """Kept from eval.py for compatibility with the original repository."""
    ner_outputs = []
    ner_targets = []
    oie_outputs = []
    oie_targets = []

    for output, target in zip(outputs, targets):
        ner_os = []
        oie_os = []
        ner_ts = []
        oie_ts = []
        output = output.replace('() ', '')
        target = target.replace('() ', '')

        for o in clean_split(output):
            if any(k in o for k in ['LOC: ', 'PER: ', 'ORG: ', 'PRO: ']):
                ner_os.append(f'({o})')
            else:
                oie_os.append(f'({o})')
        for t in clean_split(target):
            if any(k in t for k in ['LOC: ', 'PER: ', 'ORG: ', 'PRO: ']):
                ner_ts.append(f'({t})')
            else:
                oie_ts.append(f'({t})')

        ner_outputs.append(' '.join(ner_os))
        ner_targets.append(' '.join(ner_ts))
        oie_outputs.append(' '.join(oie_os))
        oie_targets.append(' '.join(oie_ts))

    object_generation_score(texts, ner_outputs, ner_targets)
    object_generation_score(texts, oie_outputs, oie_targets)


def parse_epochs(value: str):
    if re.fullmatch(r'\d+-\d+', value):
        start, end = value.split('-')
        return int(start), int(end)
    if re.fullmatch(r'\d+', value):
        epochs = int(value)
        return epochs, epochs
    raise SyntaxError(f'Option -e/--epochs received invalid argument "{value}"')


def build_training_data(task: str, data_path: str, model_name_or_path: str):
    if 'bert' in model_name_or_path.lower():
        logger.debug('Using bert model, loading IntTrainData')
        training_data = IntTrainData(data_path)
    elif task == 'glue_stsb':
        logger.debug('stsb regression, loading NumericTrainData')
        training_data = NumericTrainData(data_path)
    else:
        training_data = TrainData(data_path)

    if data_path in GLUE_DATA_PATHS:
        logger.debug('Applying task prefix %s', data_path.split('/')[-1])
        prefix = data_path.split('/')[-1].split('_')[0]
        for split in ['train', 'dev', 'test']:
            training_data.data[split] = [
                apply_prefix(prefix, x) for x in training_data.data[split]
            ]

    return training_data


def build_model(
    task: str,
    training_data,
    model_name_or_path: str,
    resume_ckpt: str | None,
    adam_epsilon: float,
    eval_batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    num_train_epochs: int,
    num_workers: int,
    train_batch_size: int,
    warmup_steps: int,
    weight_decay: float,
    gradient_checkpointing: bool,
    srl_role_token_by_canonical: dict | None,
):
    if 'bert' in model_name_or_path.lower():
        if resume_ckpt:
            model = felflarebert.load_from_checkpoint(resume_ckpt)
        else:
            model = felflarebert(
                adam_epsilon=adam_epsilon,
                eval_batch_size=eval_batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                model=model_name_or_path,
                num_train_epochs=num_train_epochs,
                num_workers=num_workers,
                train_batch_size=train_batch_size,
                warmup_steps=warmup_steps,
                weight_decay=weight_decay,
                num_labels=training_data.num_labels,
            )
        logger.info('felflare bert model being used')
        return model

    if task == 'glue_stsb':
        model = PRT5Numeric(
            adam_epsilon=adam_epsilon,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            model=model_name_or_path,
            num_train_epochs=num_train_epochs,
            num_workers=num_workers,
            train_batch_size=train_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
        )
        if resume_ckpt:
            # Preserve the original train.py behavior: initialize the T5 body
            # from the checkpoint while leaving the regression head task-specific.
            pr_model = PRT5.load_from_checkpoint(resume_ckpt)
            model.model.load_state_dict(pr_model.state_dict(), strict=False)
        return model

    if resume_ckpt:
        model = PRT5.load_from_checkpoint(resume_ckpt)
        if gradient_checkpointing:
            model.model.gradient_checkpointing_enable()
            model.model.config.use_cache = False
        return model

    return PRT5(
        adam_epsilon=adam_epsilon,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        model=model_name_or_path,
        num_train_epochs=num_train_epochs,
        num_workers=num_workers,
        train_batch_size=train_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_checkpointing=gradient_checkpointing,
        srl_role_token_by_canonical=srl_role_token_by_canonical,
    )


def generate_test_outputs(
    model,
    training_data,
    max_seq_length: int,
    eval_batch_size: int,
    num_workers: int,
):
    dl_kw = dict(
        split='test',
        tokenizer=model.tokenizer,
        max_seq_length=max_seq_length,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    if isinstance(model, PRT5) and model._srl_role_map():
        dl_kw['target_text_transform'] = model.srl_adapt_target

    test_loader = training_data.loader(**dl_kw)

    logger.info('Generating test outputs from the model that just finished training.')
    model.eval()
    with torch.no_grad():
        texts, outputs, targets = model.generate(test_loader)

    return texts, outputs, targets


def save_predictions(path: str, task: str, texts, outputs, targets):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for text, output, target in zip(texts, outputs, targets):
            if task == 'glue_stsb':
                output = float(output)
                target = float(target)
            json.dump(
                {'text': text, 'output': output, 'target': target},
                f,
                ensure_ascii=False,
            )
            f.write('\n')
    logger.info('Saved predictions to %s', path)


def evaluate(task: str, texts, outputs, targets, strict: bool):
    logger.info('Evaluating %s score.', task)

    # This intentionally matches the behavior of the eval.py you provided:
    # GLUE classification tasks, including CoLA, are scored with glueAccuracy.
    if task in GLUE_CLASSIFICATION_TASKS:
        accuracy = import_module('tasks.glueAccuracy').score(texts, outputs, targets)
        print(
            f"\n=============== Evaluation Result ===============\n"
            f"Task: {task}\n"
            f"Accuracy: {accuracy}\n"
            f"===================================================\n"
        )
        return {'accuracy': accuracy}

    if task == 'glue_stsb':
        outputs = [float(o) for o in outputs]
        targets = [float(t) for t in targets]
        pearson = import_module('tasks.' + task).score(texts, outputs, targets)
        print(
            f"\n=============== Evaluation Result ===============\n"
            f"Task: {task}\n"
            f"Pearson: {pearson}\n"
            f"===================================================\n"
        )
        return {'pearson': pearson}

    p, r, f1 = import_module('tasks.' + task).score(
        texts, outputs, targets, strict=strict
    )
    print(
        f"\n=============== Evaluation Result ===============\n"
        f"Task: {task}\n"
        f"Precision: {p}\n"
        f"Recall: {r}\n"
        f"F1: {f1}\n"
        f"===================================================\n"
    )
    return {'precision': p, 'recall': r, 'f1': f1}


def run(
    task: str,
    run_name: str,
    data_path: str,
    min_epochs: int,
    max_epochs: int,
    model_name_or_path: str = 'google-t5/t5-base',
    resume_ckpt: str | None = None,
    seed: int = 42,
    strict: bool = False,
    accelerator: str = 'gpu',
    devices: int = 1,
    precision: str = 'bf16-mixed',
    adam_epsilon: float = 1e-8,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    max_seq_length: int = 512,
    num_workers: int = 4,
    accumulate_grad_batches: int = 8,
    log_every_n_steps: int = 10,
    gradient_checkpointing: bool = False,
    srl_role_token_by_canonical: dict | None = None,
    output_dir: str = 'outputs',
    prediction_jsonl: str | None = None,
):
    """Fine-tune a model in memory, then immediately evaluate it on the test split."""
    torch.set_float32_matmul_precision('medium')
    set_seed(seed)

    logger.info('Run name: %s', run_name)
    logger.info('Task: %s', task)
    logger.info('Seed: %d', seed)
    logger.info('Training data: %s', data_path)
    logger.info('Starting model/path: %s', model_name_or_path)
    logger.info('Fine-tuning checkpoint: %s', resume_ckpt or 'None')

    training_data = build_training_data(task, data_path, model_name_or_path)
    model = build_model(
        task=task,
        training_data=training_data,
        model_name_or_path=model_name_or_path,
        resume_ckpt=resume_ckpt,
        adam_epsilon=adam_epsilon,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        num_train_epochs=max_epochs,
        num_workers=num_workers,
        train_batch_size=train_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_checkpointing=gradient_checkpointing,
        srl_role_token_by_canonical=srl_role_token_by_canonical,
    )

    model.store_data(training_data)
    logger.info('Initialized model and attached training data.')

    trainer = Trainer(
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=TensorBoardLogger(
            save_dir=os.path.join(output_dir, 'logs'),
            name=run_name,
            version=f'seed-{seed}',
        ),
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        default_root_dir=output_dir,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_checkpointing=False,  # Important: do not save Lightning checkpoints.
    )

    logger.info('Starting fine-tuning.')
    trainer.fit(model)
    logger.info('Fine-tuning complete. No checkpoint was saved.')

    texts, outputs, targets = generate_test_outputs(
        model=model,
        training_data=training_data,
        max_seq_length=max_seq_length,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    if prediction_jsonl:
        save_predictions(prediction_jsonl, task, texts, outputs, targets)

    metrics = evaluate(task, texts, outputs, targets, strict=strict)
    print(f'Seed: {seed}')
    return metrics


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'task',
        type=str,
        help='The fine-tuning/evaluation task to perform.',
    )
    parser.add_argument(
        '-m', '--model-name-or-path',
        type=str,
        default='google-t5/t5-base',
        help='Base model name/path to use.',
    )
    parser.add_argument(
        '-n', '--run-name', '--model-name', '--ckpt-name',
        dest='run_name',
        type=str,
        required=True,
        help='Name for this run/logging. No checkpoint is created.',
    )
    parser.add_argument(
        '-d', '--dataset-jsonl',
        type=str,
        help='Training/evaluation jsonl. If omitted, use the task default.',
    )
    parser.add_argument(
        '-e', '--epochs',
        type=str,
        default='3',
        help='Number of epochs. Example: 3 or 2-5.',
    )
    parser.add_argument(
        '-r', '--resume-ckpt',
        type=str,
        help=(
            'Checkpoint whose weights should initialize fine-tuning. '
            'If omitted, training starts from --model-name-or-path.'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for this fine-tuning run.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Use the strict task metric where supported.',
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=4,
        help='Training batch size per device.',
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=4,
        help='Generation/evaluation batch size.',
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=512,
        help='Max sequence length for tokenization.',
    )
    parser.add_argument(
        '--accumulate-grad-batches',
        type=int,
        default=8,
        help='Gradient accumulation steps.',
    )
    parser.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing on T5 during training.',
    )
    parser.add_argument(
        '--srl-adapt-role-tokens',
        action='store_true',
        help='SRL only: enable the repository role-token adaptation.',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader workers. Use 0 to avoid tokenizer fork warnings.',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='bf16-mixed',
    )
    parser.add_argument(
        '--accelerator',
        type=str,
        default='gpu',
    )
    parser.add_argument(
        '--devices',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--log-every-n-steps',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Used for TensorBoard logs only; checkpoints are disabled.',
    )
    parser.add_argument(
        '--prediction-jsonl',
        type=str,
        default=None,
        help='Optional path to save generated test predictions. No caching is used.',
    )

    args = parser.parse_args()

    if not args.dataset_jsonl and args.task not in DEFAULT_DATA_PATHS:
        raise NotImplementedError(args.task)

    data_path = args.dataset_jsonl or DEFAULT_DATA_PATHS[args.task]
    min_epochs, max_epochs = parse_epochs(args.epochs)

    srl_role_token_by_canonical = None
    if args.srl_adapt_role_tokens:
        if args.task != 'srl':
            raise SystemExit('--srl-adapt-role-tokens is only valid for task srl')

        from data.srl_label_tokens import (
            build_role_token_maps,
            discover_srl_roles_from_jsonl,
            resolve_srl_train_jsonl_for_roles,
        )

        train_jsonl = resolve_srl_train_jsonl_for_roles(data_path)
        roles = discover_srl_roles_from_jsonl(train_jsonl)
        srl_role_token_by_canonical, _ = build_role_token_maps(roles)
        logger.info(
            'SRL role-token adaptation enabled: %d roles from %s',
            len(roles),
            train_jsonl,
        )

    run(
        task=args.task,
        run_name=args.run_name,
        data_path=data_path,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        model_name_or_path=args.model_name_or_path,
        resume_ckpt=args.resume_ckpt,
        seed=args.seed,
        strict=args.strict,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_seq_length=args.max_seq_length,
        num_workers=args.num_workers,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        srl_role_token_by_canonical=srl_role_token_by_canonical,
        output_dir=args.output_dir,
        prediction_jsonl=args.prediction_jsonl,
    )


if __name__ == '__main__':
    main()
