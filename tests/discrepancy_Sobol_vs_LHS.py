
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt


def compute_discrepancies(d=8, sample_sizes=None, sobol_scramble=True, seed=42):
    """Compute discrepancy for LHS and Sobol over sample_sizes.

    Returns:
        sizes, disc_lhs_list, disc_sobol_list
    """
    if sample_sizes is None:
        # Powers of two are natural for Sobol
        sample_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 200000]

    disc_lhs = []
    disc_sobol = []

    for i, N in enumerate(sample_sizes):
        print(f"Computing discrepancy for N={N} ({i+1}/{len(sample_sizes)})...", end='', flush=True)
        
        # LHS
        lhs = qmc.LatinHypercube(d=d, seed=seed)
        X_lhs = lhs.random(n=N)
        disc_lhs.append(qmc.discrepancy(X_lhs, workers=1))
        
        print(f" LHS done", end='', flush=True)

        # Sobol: for Sobol we prefer powers of two; if N is not a power of two,
        # we sample the next power of two and then take the first N points.
        m = int(np.ceil(np.log2(N)))
        sobol = qmc.Sobol(d=d, scramble=sobol_scramble, seed=seed)
        X_sob = sobol.random_base2(m=m)
        # take first N rows (works if N <= 2**m)
        X_sob = X_sob[:N]
        disc_sobol.append(qmc.discrepancy(X_sob, workers=1))
        
        print(f" Sobol done ✓")

    return sample_sizes, disc_lhs, disc_sobol


def plot_discrepancies(sizes, disc_lhs, disc_sobol, out_file='discrepancy_comparison.png'):
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, disc_lhs, marker='o', label='LHS (Latin Hypercube)')
    plt.plot(sizes, disc_sobol, marker='s', label='Sobol (scrambled)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.xlabel('Number of samples (log scale)')
    plt.ylabel('Discrepancy (log scale)')
    plt.title('Discrepancy: LHS vs Sobol')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.show()


if __name__ == '__main__':
    # parameters
    d = 8
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 200000]

    sizes, dlhs, dsob = compute_discrepancies(d=d, sample_sizes=sizes)
    for s, l, sh in zip(sizes, dlhs, dsob):
        print(f"N={s:5d}  LHS disc={l:.6e}  Sobol disc={sh:.6e}")

    plot_discrepancies(sizes, dlhs, dsob)
