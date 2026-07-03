import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from pathlib import Path


try:
    import lets_be_rational.LetsBeRational as lbr
    print("Loaded via package import")

except ModuleNotFoundError:
    print("Package import failed. Loading .so manually...")

    import importlib.util
    
    so_path = Path("./venv/lib/python3.13/site-packages/lets_be_rational/_LetsBeRational.cpython-313-darwin.so")

    if not so_path.exists():
        raise FileNotFoundError(f"Could not find shared library at {so_path}")

    spec = importlib.util.spec_from_file_location("_LetsBeRational", so_path)
    lbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lbr)

    print("Loaded via direct .so import")

print("Final module:", lbr)


#### OBS FÖR WINDOWS ####

"""Behöver installera SWIG och Visual Studio Build Tools(6GB) för att köra Lets Be Rational, kan vara lite komplicerat...
I PS: 
winget install --id SWIG.SWIG

winget install --id Microsoft.VisualStudio.2022.BuildTools
    In i appen och klicka på modify. Kryssa i Desktop development with C++, i optional räcker det med MSVC v143 och Windows 11 SDK

swig -version
( Om SWIG ej hittas måste det läggas i path i terminalen innan pip körs: 
$swigDir = path till där swig ligger
$env:Path = "$env:Path;$swigDir" 
swig -version )

pip install lets_be_rational
"""

#### FÖR MAC #### kör homebrew


path = "./data/full_dataset_10000.csv"
df = pd.read_csv(path)
df.columns = df.columns.str.strip()

lm  = df["lm"].values           # lm=ln(S/K)
T   = df["T"].values
r   = df["r"].values
P   = df["price"].values        # diskonterat put-pris
P   = P * np.exp(r*T)
S   = df["S"].values
K = S / np.exp(lm)                        
q = 0.0

qflag = -1                      # Put-flagga -1 i let´s be rational bibloteket

m = lm + (r - q) * T            # m=ln(F/K)  F=S*e^((r-q)*T) --> F/K = e^ln(F/K) = e^( ln(S/K)+(r-q)*T ) = S/K*e^((r-q)*T)
F = np.exp(m) * K

# Kan även behöva kolla gränser: ( max(K-F,0), K ) samt om P är nära intrisic eller 0 --> IV = 0
"""
lower = np.maximum(K - F, 0.0) # Arbitragegränser
upper = K
eps = 1e-12

ivs_low = P <= lower + eps      # P nära intrisic --> IV = 0. "mask"
ivs_unreal = (P < lower - eps) | (P > upper + eps) # Inget reelt IV utanför detta. "mask"

ivs = np.empty(len(P), dtype=float) # tom array
ivs[ivs_low] = 0.0
ivs[ivs_unreal] = np.nan

t0 = time.perf_counter()

# Kör Let’s Be Rational
for i, (P_i, F_i, K_i, T_i) in enumerate(zip(P, F, K, T)):
    if ivs_low[i] or ivs_unreal[i]:
        continue
    ivs[i] = lbr.implied_volatility_from_a_transformed_rational_guess(P_i, F_i, K_i, T_i, qflag) #pris, forward, strike, tau & put/call som input

t1 = time.perf_counter()
elapsed = t1 - t0
print("Time elapsed for Let's Be Rational:", round(elapsed, 4))"""

ivs = df["IV"].values

#print(df["IV"].head())


### ____________ Test: IV --> Black-76 (odiskonterat) --> jämför med P_hat ____________

abs_err = np.full(len(df), np.nan, dtype=float) # tom array
P_black = np.full(len(df), np.nan, dtype=float) 

for i in range(len(df)):
    iv = df["IV"].values[i]
    if np.isfinite(iv) and iv > 0.0:
        P_black[i] = lbr.black(F[i], K[i], iv, T[i],-1)
        abs_err[i] = abs(P_black[i] - P[i])  

df["P_undisc_from_IV"] = P_black
df["P_abs_error"] = abs_err

#print(df[["lm", "tau", "r", "P_hat", "IV", "P_from_IV", "P_abs_error"]].head())


### _____________ Test Let's Be Rational vs Brent ______________

def iv_brent_lbr(P, F, K, T, lo=1e-12, hi=5.0):

    f = lambda s: lbr.black(F, K, s, T, qflag) - P
    f_lo, f_hi = f(lo), f(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        # ökar övre gränsen tills roten omsluts
        hi_exp = hi
        for _ in range(30):
            hi_exp *= 2.0
            fhi = f(hi_exp)
            if np.sign(f_lo) != np.sign(f_hi):
                hi = hi_exp
                break
        else:
            return np.nan
        
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200)

IV_brent = np.full(len(df), np.nan, dtype=float)
IV_abs_error = np.full(len(df), np.nan, dtype=float)

t2 = time.perf_counter()

#Kör Brent
for i in range(len(df)):
    IV_brent[i] = iv_brent_lbr(P[i], F[i], K[i], T[i])

t3 = time.perf_counter()
elapsed_brent = t3 - t2
print("Time elapsed for Brent:", round(elapsed_brent, 4))
#print(f"--> Let's be rational is {int(elapsed_brent/elapsed)} times faster")

for i in range(len(df)):
    IV_abs_error[i] = IV_brent[i] - ivs[i]

df["IV_brent"] = IV_brent
df["IV_abs_error"] = IV_abs_error


print(df[["lm", "T", "r", "price", "IV", "IV_brent", "P_undisc_from_IV", "P_abs_error", "IV_abs_error"]].sample(10))
print(len(df))
print(f"Antal nan-värden: {df["IV"].isna().sum()}")
print(f"Antal 0-värden: {(df["IV"] == 0).sum()}")



#### Tester som funkar med notebooken:

### ____________ Test: IV --> Black-76 (odiskonterat) --> jämför med P_hat ____________

abs_err = np.full(len(df_final), np.nan, dtype=float) # tom array
P_black = np.full(len(df_final), np.nan, dtype=float) 

for i in range(len(df_final)):
    iv_i = df_final["IV"].values[i]
    if np.isfinite(iv_i) and iv_i > 0.0:
        P_black[i] = lbr.black(F[i], K[i], iv_i, T[i],-1)
        abs_err[i] = abs(P_black[i] - P[i])  

df_final["P_from_IV"] = P_black
df_final["P_abs_error"] = abs_err

#print(df[["lm", "tau", "r", "P_hat", "IV", "P_from_IV", "P_abs_error"]].head())


### _____________ Test Let's Be Rational vs Brent ______________

def iv_brent_lbr(P, F, K, T, lo=1e-12, hi=5.0):

    f = lambda s: lbr.black(F, K, s, T, qflag) - P
    f_lo, f_hi = f(lo), f(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        # ökar övre gränsen tills roten omsluts
        hi_exp = hi
        for _ in range(30):
            hi_exp *= 2.0
            f_hi = f(hi_exp)
            if np.sign(f_lo) != np.sign(f_hi):
                hi = hi_exp
                break
        else:
            return np.nan
        
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200)

IV_brent = np.full(len(df_final), np.nan, dtype=float)
IV_abs_error = np.full(len(df_final), np.nan, dtype=float)

#Kör Brent
for i in range(len(df_final)):
    IV_brent[i] = iv_brent_lbr(P[i], F[i], K[i], T[i])

for i in range(len(df_final)):
    IV_abs_error[i] = IV_brent[i] - iv[i]

df_final["IV_brent"] = IV_brent
df_final["IV_abs_error"] = IV_abs_error


print(df_final[["lm", "T", "r", "price", "IV", "IV_brent", "P_from_IV", "P_abs_error", "IV_abs_error"]].sample(5))

