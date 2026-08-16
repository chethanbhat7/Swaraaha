import json

import torch

from model.classification import DYSFLUENCY_CLASSES
from model.evaluation.comparative_study import main, _n_params

PER_CLASS = {
    name: {
        'binary': {'f1': 0.5, 'auprc': 0.6},
        'threshold_sweep': {'best_f1': {'f1': 0.6}},
        'support': 10,
    }
    for name in DYSFLUENCY_CLASSES
}


def test_n_params_from_state_dict(tmp_path):
    path = tmp_path / 'm.pt'
    torch.save({'model_state_dict': {'w': torch.zeros(10)}}, str(path))
    assert _n_params(str(path)) == 10


def test_main_writes_report(tmp_path):
    arms_dir = tmp_path / 'arms'
    arm_dir = arms_dir / 'arm02_w2v2_multitask'
    test_dir = arm_dir / 'test'
    boli_dir = arm_dir / 'boli'
    test_dir.mkdir(parents=True)
    boli_dir.mkdir()

    def _write(dirpath, per_class, macro, tuned=None):
        report = {'per_class': per_class, 'macro_f1': macro, 'model_path': 'm.pt'}
        if tuned is not None:
            report['macro_f1_tuned'] = tuned
        (dirpath / 'multitask_report.json').write_text(json.dumps(report), encoding='utf-8')

    _write(test_dir, PER_CLASS, 0.5, 0.6)
    _write(boli_dir, PER_CLASS, 0.4, 0.45)
    (arm_dir / 'multitask_thresholds.json').write_text(
        json.dumps({'thresholds': {n: {'f1_threshold': 0.5} for n in DYSFLUENCY_CLASSES}}),
        encoding='utf-8')
    model_path = tmp_path / 'model.pt'
    torch.save({'model_state_dict': {'w': torch.zeros(10)}}, str(model_path))

    out = tmp_path / 'out'
    main(['--arms_dir', str(arms_dir),
          '--arm', f'arm02_w2v2_multitask:multitask:{model_path}',
          '--output_dir', str(out)])

    report = json.loads((out / 'comparative_study_report.json').read_text())
    arm = report['arms']['arm02_w2v2_multitask']
    assert arm['n_params'] == 10
    assert arm['test']['macro_f1'] == 0.5
    row = report['comparison'][0]
    assert row['test_macro_f1'] == 0.5
    assert row['test_macro_f1_tuned'] == 0.6
    assert row['boli_macro_f1'] == 0.4
    assert (out / 'comparative_study_report.md').exists()


def test_main_classifier_arm_aggregates_tuned(tmp_path):
    arms_dir = tmp_path / 'arms'
    arm_dir = arms_dir / 'arm01_w2v2_single'
    (arm_dir / 'test').mkdir(parents=True)
    (arm_dir / 'boli').mkdir()
    for name in DYSFLUENCY_CLASSES:
        entry = {'binary': {'f1': 0.5}, 'threshold_tuned': {'f1': 0.7}}
        (arm_dir / 'test' / f'{name}_report.json').write_text(
            json.dumps(entry), encoding='utf-8')
        (arm_dir / 'boli' / f'{name}_report.json').write_text(
            json.dumps(entry), encoding='utf-8')
    model_path = tmp_path / 'model.pt'
    torch.save({'model_state_dict': {'w': torch.zeros(10)}}, str(model_path))
    paths = ','.join(str(model_path) for _ in DYSFLUENCY_CLASSES)

    out = tmp_path / 'out'
    main(['--arms_dir', str(arms_dir),
          '--arm', f'arm01_w2v2_single:classifier:{paths}',
          '--output_dir', str(out)])

    report = json.loads((out / 'comparative_study_report.json').read_text())
    row = report['comparison'][0]
    assert row['test_macro_f1'] == 0.5
    assert row['test_macro_f1_tuned'] == 0.7
