"""
Logs the Section 4.3 benchmark to wandb (offline; upload with `wandb login` + `wandb sync wandb/offline-run-*`).

  1. one run per training run that was trained before wandb logging existed (no `wandb` marker in its summary):
     the per-epoch history of log.csv, the config and the validation summary
  2. one run 'test_results' with every table of the benchmark as wandb.Table (test metrics of every method and
     output, error decomposition, oracle studies, Table 4.13 re-check, post-hoc study, robust decoders)

Test metrics are only logged as tables (2), never as the history of a training run, so they cannot be used to pick models.
"""
import os
import csv
import glob
import json

from common import RESULTS, ROOT
WANDB_PROJECT = 'climatesam-section-4.3'  # same project as train.py

os.environ.setdefault('WANDB_MODE', 'offline')
import wandb


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def table(rows):
    cols = sorted({k for r in rows for k in r}, key=lambda k: [k for r in rows for k in r].index(k))
    return wandb.Table(columns=cols, data=[[r.get(c) if not isinstance(r.get(c), (list, dict)) else json.dumps(r.get(c))
                                            for c in cols] for r in rows])


def training_runs():
    marker = os.path.join(RESULTS, 'runs', '.wandb_logged.json')
    done = set(json.load(open(marker))) if os.path.exists(marker) else set()
    for run_dir in sorted(glob.glob(os.path.join(RESULTS, 'runs', '*', '*'))):
        summary_path = os.path.join(run_dir, 'summary.json')
        if not os.path.exists(summary_path):
            continue
        encoder, name = run_dir.split(os.sep)[-2:]
        summary = json.load(open(summary_path))
        key = f'{encoder}/{name}'
        # runs trained with --wandb already have their own offline run (robust decoders and det heads always do)
        if key in done or summary.get('args', {}).get('wandb') or name.startswith(('robust_decoder_', 'det_head_')):
            continue
        kind = 'decoder' if 'decoder' in name else 'learned_prompt' if 'learned_prompt' in name else \
            'det_head' if name.startswith('det_head') else 'prompter'
        run = wandb.init(project=WANDB_PROJECT, name=key, group=kind, dir=ROOT, config=summary.get('args', summary),
                         mode='offline', reinit=True, tags=['logged-after-training'])
        log = os.path.join(run_dir, 'log.csv')
        for row in (read_csv(log) if os.path.exists(log) else []):
            vals = {k: num(v) for k, v in row.items() if num(v) is not None}
            run.log(vals, step=int(vals.get('epoch', 0)))
        run.summary.update({k: v for k, v in summary.items() if not isinstance(v, (dict, list))})
        run.finish()
        done.add(key)
        with open(marker, 'w') as f:
            json.dump(sorted(done), f)
        print('logged', key, flush=True)


def test_tables():
    run = wandb.init(project=WANDB_PROJECT, name='test_results', group='tables', dir=ROOT, mode='offline', reinit=True)
    for enc_dir in sorted(glob.glob(os.path.join(RESULTS, 'eval', '*'))):
        enc = os.path.basename(enc_dir)
        rows, dec = [], []
        for path in sorted(glob.glob(os.path.join(enc_dir, '*.json'))):
            d = json.load(open(path))
            rows += d.get('rows', [])
            for cls, parts in d.get('error_decomposition', {}).items():
                dec.append({'method': os.path.basename(path)[:-5], 'class': cls, **parts})
        for path in sorted(glob.glob(os.path.join(enc_dir, 'decoder', '*', '*.json'))):
            decoder = path.split(os.sep)[-2]
            rows += [{'decoder': decoder, **r} for r in json.load(open(path)).get('rows', [])]
        if rows:
            run.log({f'test/{enc}/all_outputs': table(rows)})
        if dec:
            run.log({f'test/{enc}/error_decomposition': table(dec)})
    for path in sorted(glob.glob(os.path.join(RESULTS, '*', '*.csv')) + glob.glob(os.path.join(RESULTS, 'tables', '*.csv'))):
        rel = os.path.relpath(path, RESULTS)[:-4]
        rows = read_csv(path)
        if rows:
            run.log({f'csv/{rel}': table(rows)})
    for path in sorted(glob.glob(os.path.join(RESULTS, '06_posthoc', '*.json'))):
        d = json.load(open(path))
        run.log({f'posthoc/{os.path.basename(path)[:-5]}': table([{'variant': k, **v} for k, v in d['variants'].items()])})
    run.finish()


if __name__ == '__main__':
    import sys
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if part in ('runs', 'all'):
        training_runs()
    if part in ('tables', 'all'):
        test_tables()
    print('upload with:  wandb login  &&  wandb sync', os.path.join(ROOT, 'wandb', 'offline-run-*'))
