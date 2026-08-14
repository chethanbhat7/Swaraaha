import pytest

from model.config.defaults import DYSFLUENCY_CLASSES
from model.fingerprint import (
    CNN_CLASSIFIER_RESUME_KEYS,
    cnn_classifier_fingerprint,
    parse_cnn_classifier_fingerprint,
)


class _Args:
    data_dir = 'data/train'
    epochs = 20
    batch_size = 16
    lr = 3e-5
    n_mels = 128
    hop_length = 512
    max_length_seconds = 10.0
    hidden_dim = 128
    dropout = 0.4
    patience = 5
    warmup_steps = 500
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    seed = 42
    aggregator = 'pool'
    num_lstm_layers = 1
    num_transformer_layers = 1
    class_names = list(DYSFLUENCY_CLASSES)


def test_cnn_classifier_fingerprint_format_pool_all():
    args = _Args()
    fp = cnn_classifier_fingerprint(args)
    assert fp == ('cnnclf_aggpool_e20_b16_lr3e-5_n128_h512_ml10_hd128_d0.4_pa5_'
                  'wu500_wd0.01_ga1_s42_train_all')


def test_cnn_classifier_fingerprint_single_class():
    args = _Args()
    args.class_names = ['block']
    fp = cnn_classifier_fingerprint(args)
    assert fp == ('cnnclf_aggpool_e20_b16_lr3e-5_n128_h512_ml10_hd128_d0.4_pa5_'
                  'wu500_wd0.01_ga1_s42_train_block')


def test_cnn_classifier_fingerprint_aggregator_shorts():
    args = _Args()
    args.aggregator = 'lstm'
    args.num_lstm_layers = 2
    assert cnn_classifier_fingerprint(args).startswith('cnnclf_agglstm2_')
    args.aggregator = 'transformer'
    args.num_transformer_layers = 2
    assert cnn_classifier_fingerprint(args).startswith('cnnclf_aggtf2_')


def test_cnn_classifier_fingerprint_multi_class_subset_roundtrip():
    args = _Args()
    args.class_names = ['block', 'wordrep']
    fp = cnn_classifier_fingerprint(args)
    params = parse_cnn_classifier_fingerprint(fp)
    assert params['class_names'] == ['block', 'wordrep']
    assert params['data_short'] == 'train'


def test_parse_cnn_classifier_fingerprint_aggregator_roundtrip():
    args = _Args()
    args.aggregator = 'lstm'
    args.num_lstm_layers = 2
    params = parse_cnn_classifier_fingerprint(cnn_classifier_fingerprint(args))
    assert params['aggregator'] == 'lstm'
    assert params['num_lstm_layers'] == 2
    assert params['num_transformer_layers'] == 1
    args.aggregator = 'transformer'
    args.num_transformer_layers = 2
    params = parse_cnn_classifier_fingerprint(cnn_classifier_fingerprint(args))
    assert params['aggregator'] == 'transformer'
    assert params['num_transformer_layers'] == 2
    assert params['num_lstm_layers'] == 1


def test_parse_cnn_classifier_fingerprint_roundtrip():
    args = _Args()
    fp = cnn_classifier_fingerprint(args)
    params = parse_cnn_classifier_fingerprint(fp)
    assert params['epochs'] == 20
    assert params['batch_size'] == 16
    assert params['lr'] == 3e-5
    assert params['n_mels'] == 128
    assert params['hop_length'] == 512
    assert params['max_length_seconds'] == 10.0
    assert params['hidden_dim'] == 128
    assert params['dropout'] == 0.4
    assert params['seed'] == 42
    assert params['aggregator'] == 'pool'
    assert params['num_lstm_layers'] == 1
    assert params['num_transformer_layers'] == 1
    assert params['class_names'] == DYSFLUENCY_CLASSES


def test_parse_cnn_classifier_fingerprint_invalid():
    with pytest.raises(ValueError):
        parse_cnn_classifier_fingerprint('not_a_fingerprint')


def test_cnn_classifier_resume_keys_cover_args():
    args = _Args()
    assert all(hasattr(args, key) for key in CNN_CLASSIFIER_RESUME_KEYS)
