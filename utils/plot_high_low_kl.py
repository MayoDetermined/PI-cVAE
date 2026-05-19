"""
Locate diagnostics .npz, select top-N and bottom-N KL samples and run
`utils/visualize_pcvae.py` to save GT / Prior / Posterior figures for them.

Usage:
    python utils/plot_high_low_kl.py --diag_dir figs_PCVAE --top_n 8

The script will create `out_dir/high_kl` and `out_dir/low_kl` and call the
existing `utils/visualize_pcvae.py` script for the selected sample indices.
"""
import argparse
import os
import re
import subprocess
from pathlib import Path
import numpy as np


def find_latest_npz(dirpath):
    p = Path(dirpath)
    files = list(p.glob('diagnostics_epoch_*.npz'))
    if not files:
        raise FileNotFoundError(f'No diagnostics_epoch_*.npz found in {dirpath}')

    def epoch_of(pth):
        m = re.search(r'epoch_(\d+)\.npz$', pth.name)
        return int(m.group(1)) if m else -1

    files.sort(key=epoch_of)
    return str(files[-1])


def load_kld(npz_path):
    npz = np.load(npz_path, allow_pickle=True)
    print('NPZ keys:', npz.files)
    if 'kld_per_sample' in npz.files:
        kld = npz['kld_per_sample']
    else:
        keys = [k for k in npz.files if 'kld' in k.lower()]
        if keys:
            kld = npz[keys[0]]
        else:
            # Attempt compute from stored mu/logvar arrays
            if all(x in npz.files for x in ('mu_q', 'logvar_q', 'mu_c', 'logvar_c')):
                mu_q = npz['mu_q']; lv_q = npz['logvar_q']
                mu_c = npz['mu_c']; lv_c = npz['logvar_c']
                kld_per_dim = 0.5 * (lv_c - lv_q + (np.exp(lv_q) + (mu_q - mu_c) ** 2) / np.exp(lv_c) - 1.0)
                kld = kld_per_dim.sum(axis=1)
            else:
                raise KeyError('No kld_per_sample-like arrays found and cannot compute KLD from mu/logvar')
    return np.asarray(kld)


def run_visualize(samples, out_dir, checkpoint=None):
    if len(samples) == 0:
        print('No samples to visualise')
        return
    cmd = ['python', 'utils/visualize_pcvae.py', '--sample'] + [str(int(i)) for i in samples] + ['--out_dir', out_dir]
    if checkpoint:
        cmd += ['--checkpoint', checkpoint]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag_dir', default='figs_PCVAE')
    parser.add_argument('--diag_npz', default=None)
    parser.add_argument('--top_n', type=int, default=8, help='number of high/low KL samples to visualise')
    parser.add_argument('--checkpoint', default=None, help='optional checkpoint to pass to visualizer')
    parser.add_argument('--out_dir', default='figs_PCVAE/high_low_kl')
    args = parser.parse_args()

    diag_npz = args.diag_npz or find_latest_npz(args.diag_dir)
    print('Using diagnostics file:', diag_npz)
    kld = load_kld(diag_npz)
    n = min(args.top_n, len(kld))
    order = np.argsort(kld)
    low_idx = order[:n]
    high_idx = order[::-1][:n]

    print(f'selected top {n} high-KL indices: {high_idx.tolist()}')
    print(f'selected top {n} low-KL indices:  {low_idx.tolist()}')
    print('corresponding KL values (high):', kld[high_idx].tolist())
    print('corresponding KL values (low):', kld[low_idx].tolist())

    out_high = os.path.join(args.out_dir, 'high_kl')
    out_low = os.path.join(args.out_dir, 'low_kl')
    os.makedirs(out_high, exist_ok=True)
    os.makedirs(out_low, exist_ok=True)

    # Run visualizer for high-KL then low-KL
    run_visualize(high_idx, out_high, checkpoint=args.checkpoint)
    run_visualize(low_idx, out_low, checkpoint=args.checkpoint)

    print('\nFinished. Plots saved to:')
    print('  ', out_high)
    print('  ', out_low)


if __name__ == '__main__':
    main()
