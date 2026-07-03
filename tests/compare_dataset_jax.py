

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from heston_project.pricing.COS_with_jax_gradients import price_jit, grad_price


def safe_compute(row, N, L):
    try:
        params = jnp.array([
            float(row['theta']),
            float(row['v0']),
            float(row['sigma']),
            float(row['rho']),
            float(row['kappa'])
        ], dtype=jnp.float64)
        x = float(row['lm'])
        T = float(row['tau'])
        r = float(row['r'])
        price = price_jit(params, x, T, r, 0.0, N, L)
        grads = grad_price(params, x, T, r, 0.0, N, L)
        price_f = float(price)
        grads_f = np.array(grads).astype(float)
        return price_f, grads_f
    except Exception:
        return np.nan, np.array([np.nan]*5)


def main(n=500, N=256*2*2*2*2*2, L=20, out='compare_jax.csv'):
    base = Path(__file__).resolve().parents[0]

    base_path = base /"./data/dataset_100K_nofeller.csv"
    fallback_path = Path("./data/dataset_100K_nofeller.csv")

    data_path = base_path if base_path.exists() else fallback_path

    df = pd.read_csv(data_path)
    n = min(n, len(df))
    rows = []
    for i in range(n):
        row = df.iloc[i]
        model_price, model_grads = safe_compute(row, int(N), int(L))
        # price_jit returns the discounted price V = exp(-rT) * I
        # dataset `P_hat` is the undiscounted price I, so convert model outputs
        if np.isfinite(model_price):
            try:
                r = float(row['r'])
                T = float(row['tau'])
                discount_factor = np.exp(r * T)
                """model_price = float(model_price) * discount_factor
                model_grads = np.array(model_grads).astype(float) * discount_factor"""
            except Exception:
                # if conversion fails, leave as-is
                pass

        # dataset order: lm,r,tau,theta,sigma,rho,kappa,v0,P_hat,diff wrt lm,diff wrt r,...
        P_hat = float(row['p_hat'])

        # dataset gradient columns mapping to our grads order [theta, v0, sigma, rho, kappa]
        ds_theta = float(row['diff wrt theta'])
        ds_sigma = float(row['diff wrt sigma'])
        ds_rho = float(row['diff wrt rho'])
        ds_kappa = float(row['diff wrt kappa'])
        ds_v0 = float(row['diff wrt v0'])

        dataset_grads = np.array([ds_theta, ds_v0, ds_sigma, ds_rho, ds_kappa], dtype=float)

        row_out = {
            'idx': i,
            'P_hat': P_hat,
            'P_model': model_price,
            'price_err': np.nan if not np.isfinite(model_price) else (model_price - P_hat),
            'price_rel_err': np.nan if not np.isfinite(model_price) else (model_price - P_hat) / (P_hat if P_hat != 0 else 1.0),
        }

        for j, name in enumerate(['dtheta', 'dv0', 'dsigma', 'drho', 'dkappa']):
            mg = model_grads[j]
            dg = dataset_grads[j]
            row_out[f'{name}_model'] = float(mg) if np.isfinite(mg) else np.nan
            row_out[f'{name}_data'] = float(dg)
            row_out[f'{name}_err'] = np.nan if not np.isfinite(mg) else (mg - dg)
            row_out[f'{name}_rel_err'] = np.nan if not np.isfinite(mg) else (mg - dg) / (dg if dg != 0 else 1.0)

        rows.append(row_out)

    out_df = pd.DataFrame(rows)

    # Compute MSEs over rows where both model and data are finite
    metrics = {}
    mask_price = np.isfinite(out_df['P_model'])
    if mask_price.any():
        metrics['mse_price'] = np.mean((out_df.loc[mask_price, 'P_model'] - out_df.loc[mask_price, 'P_hat'])**2)
    else:
        metrics['mse_price'] = np.nan

    for name in ['dtheta', 'dv0', 'dsigma', 'drho', 'dkappa']:
        col_model = f'{name}_model'
        col_data = f'{name}_data'
        mask = np.isfinite(out_df[col_model])
        if mask.any():
            metrics[f'mse_{name}'] = np.mean((out_df.loc[mask, col_model] - out_df.loc[mask, col_data])**2)
        else:
            metrics[f'mse_{name}'] = np.nan

    # Save per-row comparison
    out_path = Path(out)
    out_df.to_csv(out_path, index=False)

    print('MSE summary:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.6e}' if np.isfinite(v) else f'  {k}: nan')

    print(f'Per-row comparison saved to: {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--N', type=int, default=256*2*2*2*2*2*2*2*2)
    p.add_argument('--L', type=int, default=12)
    p.add_argument('--out', type=str, default='compare_jax.csv')
    args = p.parse_args()
    main(n=args.n, N=args.N, L=args.L, out=args.out)
