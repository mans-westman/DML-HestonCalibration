
from heston_project.pricing.COS_nojax import *

path = "./data/dataset_100K_nofeller.csv" # Kör antingen feller eller nofeller
df = pd.read_csv(path)

df = df.sample(frac=1, random_state=42).reset_index(drop=True) # alltid samma shuffle

# df.head()

df_subset = df.iloc[500:1_000].copy()

df.columns = df_subset.columns.str.strip()

# print("Detected Columns:", df_subset.columns.tolist())
print("|-----------------------------------------|")
print(f"--- Processing {len(df_subset)} Options ---")

K_vec = np.ones(len(df_subset))             # Möjliggör konversion
S_vec = np.exp(df_subset['lm'].values)
T_vec = df_subset['tau'].values
flag_vec = np.zeros(len(df_subset))         # Puts


prices_discounted,stability = heston_cos_vectorized(
    x=df_subset['lm'].values,
    K=K_vec,
    T=T_vec,
    kappa=df_subset['kappa'].values,
    v0=df_subset['v0'].values,
    theta=df_subset['theta'].values,
    rho=df_subset['rho'].values,
    sigma=df_subset['sigma'].values,
    r=df_subset['r'].values,
    q=0.0,                                  # no dividends
    N = 512*2*2*2*2*2*2*2*2,                           # HÖG N 
    L = 12,                                 # LÅG L
    flag= flag_vec
)


df_subset['Model_Undiscounted'] = prices_discounted * np.exp(df_subset['r'] * df_subset['tau'])


df_subset['Err_Undisc'] = df_subset['Model_Undiscounted'] - df_subset['P_hat']

mse_undisc = np.mean(df_subset['Err_Undisc']**2)

'''
print("\n--- Results (First 5 Rows) ---")
print(df_subset[['lm', 'P_hat', 'Model_Undiscounted', 'Err_Undisc']].head().to_string())
mse_disc = np.mean(df_subset['Err_Disc']**2)
mse_undisc = np.mean(df_subset['Err_Undisc']**2)

'''

print('------BAD RES------')
mask = df_subset['Err_Undisc'] > 10e-7
print(df_subset.loc[mask])

print('Number of unstable entries:')
print(len(stability[stability == False]))

print("\n--- Summary Statistics ---")
print(f"MSE (Assuming Undiscounted Put): {mse_undisc:.2e}")


print("\n--- Filtering Unstable Rows & Recomputing ---")

df_stable = df_subset.loc[stability]

mse_undisc_stable = np.mean(df_stable['Err_Undisc']**2)

print(f"Original Row Count:     {len(df_subset)}")
print(f"Stable:                 {len(df_stable)}")
print(f"Unstable:               {len(df_subset) - len(df_stable)}")

print('MSE STABLE:')
print(f"MSE:                    {mse_undisc_stable:.2e}")
print('-----------')
print('MSE UNSTABLE:')
print(f"MSE:                    {df_subset.loc[~stability, 'Err_Undisc'].pow(2).mean()}")
print('-----------')
print('NAN COUNT:')
print(df_subset['Model_Undiscounted'].isna().sum())


'''
np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=6)

print(df.loc[343])
print(df.loc[849])
'''

abs_err = np.abs(df_subset['Err_Undisc'])
max_err = np.max(abs_err)
max_err_idx = np.argmax(abs_err)
p99_err = np.percentile(abs_err, 99)

print('----------')

print(f"Max Absolute Error:     {max_err:.2e}")
print(f"99th Percentile Error:  {p99_err:.2e}")
print(f"Worst Case Index:       {max_err_idx}")

worst_case = df_subset.iloc[max_err_idx]
print("\n[Worst Case Parameters]")
print(worst_case[['lm', 'tau', 'kappa', 'v0', 'sigma', 'rho']].to_string())


print("\n[Tail Behavior Check]")

otm_mask = df_subset['lm'] > 0.5
if otm_mask.any():
    otm_prices = prices_discounted[otm_mask]
    avg_otm_price = np.mean(otm_prices)
    neg_prices = np.sum(otm_prices < -1e-9) # Allow small float noise
    print(f"Avg Deep OTM Price:     {avg_otm_price:.2e} (Should be close to 0)")
    print(f"Negative Prices Found:  {neg_prices} (Should be 0)")
else:
    print("No Deep OTM options in this subset.")

itm_mask = df_subset['lm'] < -0.5
if itm_mask.any():
    itm_prices = prices_discounted[itm_mask]
    
    S_itm = np.exp(df_subset.loc[itm_mask, 'lm'].values)
    K_itm = 1.0
    r_itm = df_subset.loc[itm_mask, 'r'].values
    T_itm = df_subset.loc[itm_mask, 'tau'].values
    
    intrinsic_put = K_itm * np.exp(-r_itm * T_itm) - S_itm
    
    itm_deviation = np.abs(itm_prices - intrinsic_put)
    print(f"Max ITM Deviation:      {np.max(itm_deviation):.2e} (Should be small, dominated by Time Value)")
else:
    print("No Deep ITM options in this subset.")



print("\n|-------------|")
if max_err < 1e-4 and neg_prices == 0:
    print("OK!")
else:
    print("RES: ❌")
print("|---------------|")