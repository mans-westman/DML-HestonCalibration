# Parameter bounds
param_bounds = {
    "lm":    [-1.5, 1.5],
    "r":     [-0.01, 0.10],
    "T":     [1/52, 3.0],
    "theta": [0.0, 1.0],
    "sigma": [0.1, 1.5],
    "rho":   [-0.95, 0.0],
    "kappa": [0.05, 5.0],
    "v0":    [0.0, 1.0],
}


# Min-max scaling formula
def min_max_scale(x, xmin, xmax, zero_centered):
    # Scales to [-1,1]
    if zero_centered:
        return 2 * (x - xmin) / (xmax - xmin) - 1
    # Scales to [0,1]
    else: 
        return (x - xmin) / (xmax - xmin)
    
# Min-max scaling for inputs 
def scale_inputs(data, zero_centered):

    data_c = data.copy()

    for k, (kmin, kmax) in param_bounds.items():
        
        data_c[k] = min_max_scale(data_c[k], kmin, kmax, zero_centered)
    
    return data_c
    
# Min-max scaling for price
def scale_price(series, zero_centered):
    pmin = series.min()
    pmax = series.max()

    return min_max_scale(series, pmin, pmax, zero_centered)