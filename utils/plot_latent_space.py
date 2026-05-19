"""
Plot latent space from diagnostics .npz produced by `utils/visualize_pcvae.py`.
- Loads mu_q, mu_c, optional kld_per_sample from diagnostics npz
- Produces PCA scatter (mu_q points) with mu_c overlay and color by metric
- Optionally runs t-SNE on a subsample if `--tsne` provided and sklearn available

Usage:
    python utils/plot_latent_space.py --diag_npz figs_PCVAE/diagnostics_epoch_209.npz --out_dir figs_PCVAE --color_by kld

"""
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    SKLEARN_TSNE = True
except Exception:
    SKLEARN_TSNE = False


def find_latest_npz(dirpath):
    files = list(Path(dirpath).glob('diagnostics_epoch_*.npz'))
    if not files:
        return None
    # pick highest epoch number
    def epoch_of(p):
        try:
            s = p.stem.split('_')[-1]
            return int(s)
        except Exception:
            return -1
    files.sort(key=epoch_of)
    return str(files[-1])


def pca_2d(X, n_components=2):
    # center
    Xc = X - X.mean(axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    coords = Xc.dot(comps.T)
    return coords, comps


def plot_pca(mu_q, mu_c, color_vals, out_path, title='Latent PCA', annotate_arrows=True):
    coords_q, comps = pca_2d(mu_q)
    coords_c = (mu_c - mu_q.mean(axis=0)).dot(comps.T)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(coords_q[:, 0], coords_q[:, 1], c=color_vals, s=8, cmap='viridis', alpha=0.85)
    # overlay prior centers
    plt.scatter(coords_c[:, 0], coords_c[:, 1], c='k', s=18, marker='x', alpha=0.6)
    # optional arrows for a subset
    if annotate_arrows:
        N = len(coords_q)
        m = min(600, N)
        idx = np.random.choice(N, size=m, replace=False)
        dx = coords_q[idx, 0] - coords_c[idx, 0]
        dy = coords_q[idx, 1] - coords_c[idx, 1]
        plt.quiver(coords_c[idx, 0], coords_c[idx, 1], dx, dy, angles='xy', scale_units='xy', scale=1, width=0.0015, color='gray', alpha=0.18)
    plt.colorbar(sc, label='color')
    plt.title(title)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_tsne(mu_q, mu_c, color_vals, out_path, subsample=2000, random_state=0, arrows=False):
    if not SKLEARN_TSNE:
        return None
    # combine for joint embedding (mu_q stacked with mu_c)
    N = mu_q.shape[0]
    use_idx = np.arange(N)
    if N > subsample:
        use_idx = np.random.RandomState(random_state).choice(N, size=subsample, replace=False)
        mu_q_s = mu_q[use_idx]
        mu_c_s = mu_c[use_idx]
        stack = np.vstack([mu_q_s, mu_c_s])
    else:
        stack = np.vstack([mu_q, mu_c])
        use_idx = np.arange(N)
    ts = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=random_state)
    coords = ts.fit_transform(stack)
    half = coords.shape[0] // 2
    coords_q = coords[:half]
    coords_c = coords[half:half+half]
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(coords_q[:, 0], coords_q[:, 1], c=color_vals[use_idx], s=8, cmap='viridis', alpha=0.85)
    plt.scatter(coords_c[:, 0], coords_c[:, 1], c='k', s=18, marker='x', alpha=0.6)
    # draw arrows showing displacement mu_c -> mu_q for a subset
    if arrows:
        Ncoords = coords_q.shape[0]
        m = min(600, Ncoords)
        idx = np.random.RandomState(random_state).choice(Ncoords, size=m, replace=False)
        dx = coords_q[idx, 0] - coords_c[idx, 0]
        dy = coords_q[idx, 1] - coords_c[idx, 1]
        plt.quiver(coords_c[idx, 0], coords_c[idx, 1], dx, dy, angles='xy', scale_units='xy', scale=1, width=0.0015, color='gray', alpha=0.25)
    plt.colorbar(sc, label='color')
    plt.title('Latent t-SNE (joint embedding)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag_npz', default=None)
    parser.add_argument('--out_dir', default='figs_PCVAE')
    parser.add_argument('--color_by', choices=['kld','mu_l2','n_Nplus','none'], default='kld', help='Which metric to color points by')
    parser.add_argument('--species_index', type=int, default=3, help='Species index in na array to compute integrated value (default N+)')
    parser.add_argument('--tsne', action='store_true', help='Run t-SNE (may be slow)')
    parser.add_argument('--tsne_subsample', type=int, default=2000, help='Subsample size for t-SNE (default 2000)')
    parser.add_argument('--tsne_full', action='store_true', help='Force t-SNE on full dataset (overrides subsample)')
    parser.add_argument('--tsne_arrows', action='store_true', help='Overlay arrows mu_c -> mu_q on t-SNE (may clutter)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diag_npz = args.diag_npz
    if diag_npz is None:
        diag_npz = find_latest_npz(str(out_dir))
        if diag_npz is None:
            raise FileNotFoundError('No diagnostics .npz provided and none found in out_dir')
    print('Using diagnostics:', diag_npz)
    d = np.load(diag_npz)
    mu_q = d['mu_q']
    mu_c = d['mu_c']
    kld = d.get('kld_per_sample', None)

    # compute mu_l2
    mu_l2 = np.linalg.norm(mu_q - mu_c, axis=1)

    # determine color values
    if args.color_by == 'kld' and kld is not None:
        color_vals = kld
        color_label = 'KL per sample'
    elif args.color_by == 'mu_l2':
        color_vals = mu_l2
        color_label = 'L2(mu_q-mu_c)'
    elif args.color_by == 'n_Nplus':
        # load test set densities
        na_path = os.path.join('a_dataset', 'test', 'na_tmp.npy')
        if not os.path.exists(na_path):
            raise FileNotFoundError(f'{na_path} not found')
        na = np.load(na_path)  # shape (N, NX, NY, NS)
        # integrated per-sample
        nvals = na[..., args.species_index].sum(axis=(1,2))
        color_vals = nvals
        color_label = f'integrated n species[{args.species_index}]'
    else:
        color_vals = np.zeros(mu_q.shape[0])
        color_label = 'none'

    # normalize color for plotting (avoid extreme skew)
    cmin, cmax = np.percentile(color_vals, [1, 99]) if len(color_vals)>0 else (0,1)
    # plot PCA
    p_out = out_dir / 'latent_space_pca.png'
    print('Plotting PCA...')
    plot_pca(mu_q, mu_c, color_vals, str(p_out), title=f'Latent PCA colored by {color_label}')
    print('Saved', p_out)

    # optionally t-SNE
    if args.tsne:
        if not SKLEARN_TSNE:
            print('sklearn TSNE not available, skipping t-SNE')
        else:
            t_out = out_dir / 'latent_space_tsne.png'
            print('Running t-SNE (may take time)...')
            subs = args.tsne_subsample
            if args.tsne_full:
                subs = mu_q.shape[0]
            plot_tsne(mu_q, mu_c, color_vals, str(t_out), subsample=subs, random_state=0, arrows=args.tsne_arrows)
            print('Saved', t_out)


if __name__ == '__main__':
    main()
