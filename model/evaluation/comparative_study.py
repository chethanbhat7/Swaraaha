"""Assemble the comparative study report from per-arm evaluation artifacts.

Arm report layout (under ``--arms_dir/<arm>``):

  multitask_thresholds.json     val-tuned per-class thresholds (arms 1-7; arm 1 and 5 merged from per-class sweeps)
  test/multitask_report.json    test-set metrics (arms 2-7)
  boli/multitask_report.json    Boli subset metrics (arms 2-7)
  test/<class>_report.json      test-set metrics (arm 1, per class)
  boli/<class>_report.json      Boli subset metrics (arm 1)
  test/<class>/multitask_report.json   test-set metrics (arm 5, per class)

``--arm`` spec format: ``name:model_type:model_path[,...]`` where
``model_type`` is one of ``classifier`` (arm 1), ``multitask`` (arms 2-4,
6-7) or ``multitask_single`` (arm 5). For ``classifier`` arms, pass the five
class model paths comma-separated in ``DYSFLUENCY_CLASSES`` order.
"""

import argparse
import json
import os

import torch

from model.classification import DYSFLUENCY_CLASSES
from model.evaluation.metrics import save_report


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _n_params(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    return int(sum(v.numel() for v in state_dict.values() if hasattr(v, 'numel')))


def _binary_f1(entry):
    return (entry.get('binary') or {}).get('f1')


def _binary_auprc(entry):
    return (entry.get('binary') or {}).get('auprc')


def _tuned_f1(entry):
    return ((entry.get('threshold_tuned') or {}).get('f1')
            or (entry.get('binary') or {}).get('f1'))


def _class_entry(arm_dir, split, class_name):
    path = os.path.join(arm_dir, split, f'{class_name}_report.json')
    data = _load_json(path)
    return data.get('per_class', {}).get(class_name, data)


def _aggregate_per_class(per_class, value_fn):
    values = [value_fn(v) for v in per_class.values() if value_fn(v) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _collect_arm(arm_name, model_type, model_paths, arms_dir):
    arm_dir = os.path.join(arms_dir, arm_name)
    entry = {'model_type': model_type, 'n_params': _n_params(model_paths[0])}
    if model_type == 'classifier':
        test = {'per_class': {}}
        boli = {'per_class': {}}
        for class_name, path in zip(DYSFLUENCY_CLASSES, model_paths):
            test['per_class'][class_name] = _class_entry(arm_dir, 'test', class_name)
            boli['per_class'][class_name] = _class_entry(arm_dir, 'boli', class_name)
        for split in (test, boli):
            split['macro_f1'] = _aggregate_per_class(split['per_class'], _binary_f1)
            split['macro_f1_tuned'] = _aggregate_per_class(split['per_class'], _tuned_f1)
        entry['test'] = test
        entry['boli'] = boli
    else:
        thresholds_path = os.path.join(arm_dir, 'multitask_thresholds.json')
        entry['thresholds'] = (_load_json(thresholds_path)
                               if os.path.exists(thresholds_path) else None)
        if model_type == 'multitask_single':
            test = {'per_class': {}}
            boli = {'per_class': {}}
            for class_name in DYSFLUENCY_CLASSES:
                test['per_class'][class_name] = _load_json(os.path.join(
                    arm_dir, 'test', class_name, 'multitask_report.json')
                )['per_class'][class_name]
                boli['per_class'][class_name] = _load_json(os.path.join(
                    arm_dir, 'boli', class_name, 'multitask_report.json')
                )['per_class'][class_name]
            for split in (test, boli):
                split['macro_f1'] = _aggregate_per_class(split['per_class'], _binary_f1)
                split['macro_f1_tuned'] = _aggregate_per_class(
                    split['per_class'], _tuned_f1)
            entry['test'] = test
            entry['boli'] = boli
        else:
            entry['test'] = _load_json(os.path.join(arm_dir, 'test', 'multitask_report.json'))
            entry['boli'] = _load_json(os.path.join(arm_dir, 'boli', 'multitask_report.json'))
    return entry


def _write_markdown(report, path):
    rows = report['comparison']
    lines = ['# Comparative Study Report', '']
    header = ['Arm', 'Type', 'Params', 'Test F1@0.5', 'Test F1@tuned',
              'Boli F1@0.5', 'Boli F1@tuned']
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '|'.join(['---'] * len(header)) + '|')
    for row in rows:
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
            row['arm'], row['model_type'], row['n_params'],
            f"{row['test_macro_f1']:.4f}" if row['test_macro_f1'] is not None else '-',
            f"{row['test_macro_f1_tuned']:.4f}" if row['test_macro_f1_tuned'] is not None else '-',
            f"{row['boli_macro_f1']:.4f}" if row['boli_macro_f1'] is not None else '-',
            f"{row['boli_macro_f1_tuned']:.4f}" if row['boli_macro_f1_tuned'] is not None else '-',
        ))
    lines.append('')
    lines.append('Honest labeling: "in-distribution held-out (same-speaker '
                 'overlap)" for test; "cross-corpus held-out" for Boli. '
                 'Single seed 42; thresholds tuned on val only.')
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms_dir', type=str, default='model/evaluation/reports/arms')
    parser.add_argument('--arm', action='append', required=True,
                        help='arm_name:model_type:model_path[,...] (repeatable)')
    parser.add_argument('--output_dir', type=str, default='model/evaluation/reports')
    args = parser.parse_args(argv)

    arms = {}
    comparison = []
    for spec in args.arm:
        parts = spec.split(':')
        arm_name, model_type = parts[0], parts[1]
        model_paths = [p for p in parts[2].split(',') if p]
        entry = _collect_arm(arm_name, model_type, model_paths, args.arms_dir)
        arms[arm_name] = entry
        test = entry['test']
        boli = entry['boli']
        comparison.append({
            'arm': arm_name,
            'model_type': model_type,
            'n_params': entry['n_params'],
            'test_macro_f1': test.get('macro_f1'),
            'test_macro_f1_tuned': test.get('macro_f1_tuned'),
            'test_mean_auprc': (_aggregate_per_class(test['per_class'], _binary_auprc)
                                if test.get('per_class') else None),
            'boli_macro_f1': boli.get('macro_f1'),
            'boli_macro_f1_tuned': boli.get('macro_f1_tuned'),
        })

    report = {'arms': arms, 'comparison': comparison}
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, 'comparative_study_report.json')
    save_report(report, json_path)
    md_path = os.path.join(args.output_dir, 'comparative_study_report.md')
    _write_markdown(report, md_path)
    print(f'Wrote {json_path}')
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    main()
