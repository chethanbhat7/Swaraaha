import torch
from torch.utils.data import DataLoader, TensorDataset

from model.classification.cnn_multitask import CNNMultitaskClassifier
from model.config.defaults import DYSFLUENCY_CLASSES
from model.training import train_cnn_classifier as tcc
from model.training import train_multitask_classifier as tmc
from model.training.utils import FocalLoss

CLASS_NAMES = list(DYSFLUENCY_CLASSES)


def test_parse_args_cnn_defaults():
    args = tcc.parse_args([])
    assert args.aggregator == 'pool'
    assert args.n_mels == 128
    assert args.hop_length == 512
    assert args.n_fft == 2048
    assert args.hidden_dim == 128
    assert args.dropout == 0.4
    assert args.num_lstm_layers == 1
    assert args.num_transformer_layers == 1
    assert args.class_names == CLASS_NAMES


def test_parse_args_single_class():
    args = tcc.parse_args(['--class_names', 'block'])
    assert args.class_names == ['block']


def test_parse_args_n_fft_override():
    args = tcc.parse_args(['--n_fft', '1024'])
    assert args.n_fft == 1024


def test_train_one_epoch_with_subset_class_names():
    model = CNNMultitaskClassifier(n_mels=8, hidden_dim=8, class_names=['block'],
                                   aggregator='pool')
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(
        torch.randn(4, 1, 8, 8),
        torch.zeros(4, 5, dtype=torch.float32),
    ), batch_size=2)
    loss = tmc.train_one_epoch(model, loader, optimizer, None, criterion,
                               torch.device('cpu'), class_names=['block'])
    assert loss > 0


def test_strip_compile_prefix_when_live_uncompiled():
    ckpt = {'_orig_mod.heads.block.weight': torch.ones(2, 8),
            '_orig_mod.heads.block.bias': torch.zeros(2)}
    live = {'heads.block.weight': torch.zeros(2, 8),
            'heads.block.bias': torch.ones(2)}
    stripped = tcc._strip_compile_prefix_if_needed(ckpt, live)
    assert '_orig_mod.heads.block.weight' not in stripped
    assert stripped['heads.block.weight'].equal(torch.ones(2, 8))


def test_cnn_trainer_default_is_class_balanced(tmp_path):
    from model.training.train_cnn_classifier import parse_args

    args = parse_args([])
    assert args.class_balanced is True
