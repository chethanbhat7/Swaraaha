"""Train a multitask CNN classifier (comparative-study arms 4-7).

Supports three sequence aggregators on top of a shared convolutional encoder:

  --aggregator pool        global mean-pool (arm 4, ablation candidate)
  --aggregator lstm        single-layer LSTM (arm 6)
  --aggregator transformer TransformerEncoder (arm 7)

Single-head arms (arm 5) use the same script with ``--class_names <one class>``.

Usage:
    python -m model.training.train_cnn_classifier --aggregator pool
"""

import argparse
import logging
import os
import sys
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.classification.cnn_multitask import CNNMultitaskClassifier
from model.config.defaults import AUGMENTATION_ENABLED, DYSFLUENCY_CLASSES
from model.data.augmentation import AugmentedDataset, SpectrogramAugmentor
from model.data.dataset import SpectrogramClassificationDataset
from model.fingerprint import CNN_CLASSIFIER_RESUME_KEYS, cnn_classifier_fingerprint
from model.training.train_multitask_classifier import (
    SubsetDataset,
    evaluate_multitask,
    multitask_loss,
    set_seed,
    stratified_split,
    train_one_epoch,
)
from model.training.utils import (
    CSVLogger,
    EarlyStopping,
    FocalLoss,
    TeeLogger,
    get_warmup_linear_schedule,
    maybe_skip_completed,
    save_resume_state,
    try_load_resume,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train a multitask CNN classifier')
    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--output_dir', type=str, default='model/weights')
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--n_fft', type=int, default=2048,
                        help='FFT window size for the mel-spectrogram.')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--aggregator', type=str, default='pool',
                        choices=['pool', 'lstm', 'transformer'])
    parser.add_argument('--num_lstm_layers', type=int, default=1)
    parser.add_argument('--num_transformer_layers', type=int, default=1)
    parser.add_argument('--max_length_seconds', type=float, default=3.0)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--class_names', type=str, default=','.join(DYSFLUENCY_CLASSES))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['focal', 'cross_entropy'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory for preprocessed audio (auto-derived from data_dir if omitted).')
    parser.add_argument('--clean', action='store_true',
                        help='Start training from scratch (ignore resume checkpoint)')
    args = parser.parse_args(argv)
    args.class_names = [c.strip() for c in args.class_names.split(',') if c.strip()]
    return args


def _strip_compile_prefix_if_needed(state_dict, live_state_dict):
    """Strip the ``_orig_mod.`` torch.compile prefix from checkpoint keys when
    the live model is uncompiled (cross-device resume, e.g. CUDA->CPU)."""
    has_prefix = any(k.startswith('_orig_mod.') for k in state_dict)
    live_has_prefix = any(k.startswith('_orig_mod.') for k in live_state_dict)
    if has_prefix and not live_has_prefix:
        return {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
    return state_dict


def train(args):
    set_seed(args.seed)
    fp = cnn_classifier_fingerprint(args)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tee = TeeLogger(os.path.join(output_dir, f'{fp}_training.log'))
    sys.stdout = tee

    print(f'Running fingerprint: {fp}')
    print('Config:')
    for key in CNN_CLASSIFIER_RESUME_KEYS:
        print(f'  {key}: {getattr(args, key, None)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    num_workers = args.num_workers or (os.cpu_count() or 4)

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(args.data_dir.rstrip('/')),
            'cache',
            os.path.basename(args.data_dir.rstrip('/')),
        )
    dataset = SpectrogramClassificationDataset(
        args.data_dir,
        sr=16000,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        max_length_seconds=args.max_length_seconds,
        cache_dir=cache_dir,
    )
    print(f'Cache: {cache_dir}')
    if len(dataset) == 0:
        print(f'No samples found in {args.data_dir}. '
              'Prepare the dataset first (see model/data/setup.py).')
        sys.exit(1)

    train_idx, val_idx = stratified_split(dataset, val_ratio=0.2, seed=args.seed)
    train_dataset = SubsetDataset(dataset, train_idx)
    val_dataset = SubsetDataset(dataset, val_idx)
    if AUGMENTATION_ENABLED:
        train_dataset = AugmentedDataset(
            train_dataset,
            augmentor=None,
            augment_spectrogram=True,
            spectrogram_augmentor=SpectrogramAugmentor(),
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    criterion = (FocalLoss(gamma=args.focal_gamma) if args.loss_type == 'focal'
                 else nn.CrossEntropyLoss())
    model = CNNMultitaskClassifier(
        n_mels=args.n_mels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        class_names=args.class_names,
        aggregator=args.aggregator,
        num_lstm_layers=args.num_lstm_layers,
        num_transformer_layers=args.num_transformer_layers,
    )
    model.model.to(device)
    if device.type == 'cuda':
        warnings.filterwarnings('ignore', category=UserWarning)
        logging.getLogger('torch._dynamo').setLevel(logging.ERROR)
        torch._dynamo.config.suppress_errors = True
        model._model = torch.compile(model._model)
    print(f'Total trainable parameters: {model.count_parameters():,}')

    resume_ckpt = try_load_resume(args, device, fp)
    skip_history = maybe_skip_completed(resume_ckpt, args.epochs)
    if skip_history is not None:
        tee.close()
        return skip_history
    start_epoch = 0
    best_f1 = -float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'macro_f1': []}
    if resume_ckpt is not None:
        state_dict = _strip_compile_prefix_if_needed(
            resume_ckpt['model_state_dict'], model.model.state_dict())
        model.model.load_state_dict(state_dict)
        start_epoch = resume_ckpt['epoch'] + 1
        best_f1 = resume_ckpt['best_f1']
        history = resume_ckpt['history']

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    optim_steps_per_epoch = ((len(train_loader) + args.gradient_accumulation_steps - 1)
                             // args.gradient_accumulation_steps)
    total_optim_steps = optim_steps_per_epoch * args.epochs
    scheduler = get_warmup_linear_schedule(optimizer, args.warmup_steps, total_optim_steps)
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])

    logger = CSVLogger(
        os.path.join(output_dir, f'{fp}_log.csv'),
        ['epoch', 'train_loss', 'val_loss', 'val_acc', 'macro_f1', 'lr'],
        mode='a' if resume_ckpt is not None else 'w',
    )
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    if start_epoch > args.epochs:
        model.save(os.path.join(output_dir, f'{fp}_final.pt'))
        tee.close()
        return history

    train_start = time.time()
    last_epoch = max(start_epoch - 1, 0)
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            accumulation_steps=args.gradient_accumulation_steps,
            class_names=args.class_names,
        )
        current_lr = optimizer.param_groups[0]['lr']
        val_acc, macro_f1, val_loss = evaluate_multitask(
            model, val_loader, device, class_names=args.class_names,
        )
        logger.log(epoch=epoch + 1, train_loss=f'{train_loss:.4f}',
                   val_loss=f'{val_loss:.4f}', val_acc=f'{val_acc:.4f}',
                   macro_f1=f'{macro_f1:.4f}', lr=f'{current_lr:.2e}')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['macro_f1'].append(macro_f1)
        epoch_sec = time.time() - epoch_start
        print(f'Epoch {epoch + 1:3d}/{args.epochs} | loss={train_loss:.4f} '
              f'| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} '
              f'| macro_F1={macro_f1:.4f} | lr={current_lr:.2e} | {epoch_sec:.1f}s')
        save_resume_state(model, optimizer, scheduler, epoch, best_f1, history, args, fp,
                          resume_keys=CNN_CLASSIFIER_RESUME_KEYS)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            model.save(os.path.join(output_dir, f'{fp}_best.pt'))
        if early_stopping.step(macro_f1):
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    model.save(os.path.join(output_dir, f'{fp}_final.pt'))
    save_resume_state(model, optimizer, scheduler, last_epoch, best_f1, history, args, fp,
                      resume_keys=CNN_CLASSIFIER_RESUME_KEYS, completed=True)
    print(f'Total training time: {time.time() - train_start:.1f}s')
    logger.close()
    tee.close()
    return history


def main(argv=None):
    args = parse_args(argv)
    train(args)


if __name__ == '__main__':
    main()
