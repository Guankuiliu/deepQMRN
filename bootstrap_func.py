import numpy as np
from utils import *
from tqdm import tqdm
from pygam import ExpectileGAM, s  # to estimate GAM quantile growth models
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from statsmodels.regression.quantile_regression import QuantReg  # to estimate exponential quantile growth models
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def bootstrap_resample(X, y, n_samples=None):
    """Generate bootstrap resample of data"""
    if n_samples is None:
        n_samples = len(X)
    idx = np.random.choice(len(X), n_samples, replace=True)
    return X[idx], y[idx]


def calculate_coverage_proportion(y_true, q10_pred, q90_pred):
    """
    Calculate the proportion of observations falling between 
    predicted 0.1 and 0.9 quantiles (should be close to 0.8)
    """
    is_inside_interval = (y_true >= q10_pred) & (y_true <= q90_pred)
    return np.mean(is_inside_interval)


def bootstrap_egm(age, size, n_bootstrap):
    """Bootstrap EGM parameters and predictions with error calculation"""
    a_boot, b_boot = [], []
    pred_boot = []
    error_metrics = {'mse': [], 'mae': [], 'mape': []}
    
    age_range = np.linspace(age.min(), age.max(), 200)
    
    for _ in range(n_bootstrap):
        age_bs, size_bs = bootstrap_resample(age, size)
        # Use non-linear least squares to fit (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
        popt, _ = curve_fit(growth_func, age_bs, size_bs)  # independent variable: age_bs; dependent variable: size_bs; model form: see 'growth_func' in utils.py
        pred = growth_func(age_bs, *popt)
        
        a_boot.append(popt[0])
        b_boot.append(popt[1])
        pred_boot.append(growth_func(age_range, *popt))
        
        # Calculate errors on bootstrap sample
        error_metrics['mse'].append(mean_squared_error(size_bs, pred))
        error_metrics['mae'].append(mean_absolute_error(size_bs, pred))
        error_metrics['mape'].append(mean_absolute_percentage_error(size_bs, pred))
    
    return (np.array(a_boot), np.array(b_boot), 
            np.array(pred_boot), error_metrics)


def bootstrap_qegm(age, size, egm_params, quantiles, n_bootstrap):
    """Bootstrap QEGM parameters and predictions with error and coverage proportions calculation"""
    a_boot = {q: [] for q in quantiles}
    b_boot = {q: [] for q in quantiles}
    pred_boot = {q: [] for q in quantiles}
    error_metrics = {q: {'mse': [], 'mae': [], 'mape': []} for q in quantiles}
    coverage_proportions = []
    
    age_range = np.linspace(age.min(), age.max(), 200)
    X = np.column_stack((age, growth_func(age, *egm_params)))
    X_range = np.column_stack((age_range, growth_func(age_range, *egm_params)))
    
    for _ in range(n_bootstrap):

        # Store predictions for 0.1 and 0.9 quantiles to calculate coverage
        q10_pred = None
        q90_pred = None
        
        age_bs, size_bs = bootstrap_resample(age, size)
        X_bs = np.column_stack((age_bs, growth_func(age_bs, *egm_params))) # independent variable: age_bs; model form: see 'growth_func' in utils.py
        
        for q in quantiles:
            # use iterative reweighted least squares to fit (https://www.statsmodels.org/dev/generated/statsmodels.regression.quantile_regression.QuantReg.html)
            qm = QuantReg(size_bs, X_bs).fit(q=q, max_iter=10000)  # dependent variable: size_bs; model form: see 'growth_func' in utils.py
            pred = qm.predict(X_bs)
            
            a_boot[q].append(qm.params[0])
            b_boot[q].append(qm.params[1])
            pred_boot[q].append(qm.predict(X_range))

            # Store 0.1 and 0.9 quantile predictions for coverage calculation
            if abs(q - 0.1) < 0.001:
                q10_pred = pred
            elif abs(q - 0.9) < 0.001:
                q90_pred = pred
            
            # Calculate errors on bootstrap sample
            error_metrics[q]['mse'].append(mean_squared_error(size_bs, pred))
            error_metrics[q]['mae'].append(mean_absolute_error(size_bs, pred))
            error_metrics[q]['mape'].append(mean_absolute_percentage_error(size_bs, pred))

        # Calculate coverage proportion for this bootstrap sample
        if q10_pred is not None and q90_pred is not None:
            coverage = calculate_coverage_proportion(size_bs, q10_pred, q90_pred)
            coverage_proportions.append(coverage)
    
    return a_boot, b_boot, pred_boot, error_metrics, coverage_proportions  # Return coverage proportions


def bootstrap_qgagm(age, size, quantiles, spline_num, n_bootstrap):
    """Bootstrap QGAGM predictions with error and coverage proportions  calculation"""
    pred_boot = {q: [] for q in quantiles}
    error_metrics = {q: {'mse': [], 'mae': [], 'mape': []} for q in quantiles}
    coverage_proportions = []
    
    age_range = np.linspace(age.min(), age.max(), 200)
    
    for _ in range(n_bootstrap):
        # Store predictions for 0.1 and 0.9 quantiles to calculate coverage
        q10_pred = None
        q90_pred = None
        
        age_bs, size_bs = bootstrap_resample(age, size)
        
        for q in quantiles:
            # Use Least Asymmetrically Weighted Squares to fit (https://pygam.readthedocs.io/en/latest/api/expectilegam.html)
            model = ExpectileGAM(s(0, n_splines=spline_num), expectile=q).fit(age_bs, size_bs)  # independent variable: age_bs; dependent variable: size_bs
            pred = model.predict(age_bs)
            pred_boot[q].append(model.predict(age_range))

            # Store 0.1 and 0.9 quantile predictions for coverage calculation
            if abs(q - 0.1) < 0.001:
                q10_pred = pred
            elif abs(q - 0.9) < 0.001:
                q90_pred = pred
            
            # Calculate errors
            error_metrics[q]['mse'].append(mean_squared_error(size_bs, pred))
            error_metrics[q]['mae'].append(mean_absolute_error(size_bs, pred))
            error_metrics[q]['mape'].append(mean_absolute_percentage_error(size_bs, pred))

        # Calculate coverage proportion
        if q10_pred is not None and q90_pred is not None:
            q10_pred = np.squeeze(q10_pred)
            q90_pred = np.squeeze(q90_pred)
            coverage = calculate_coverage_proportion(size_bs, q10_pred, q90_pred)
            coverage_proportions.append(coverage)
    
    return pred_boot, error_metrics, coverage_proportions


def bootstrap_deepqgm(age, size, quantiles, n_bootstrap):
    """Bootstrap DeepQGM predictions with error and coverage proportions  calculation"""
    pred_boot = {q: [] for q in quantiles}
    error_metrics = {q: {'mse': [], 'mae': [], 'mape': []} for q in quantiles}
    coverage_proportions = []
    
    age_range = np.linspace(age.min(), age.max(), 200).reshape(-1, 1)
    
    for _ in tqdm(range(n_bootstrap), desc="DeepQGM Bootstrapping"):
        # Store predictions for 0.1 and 0.9 quantiles to calculate coverage
        q10_pred = None
        q90_pred = None
        
        age_bs, size_bs = bootstrap_resample(age, size)
        
        # Train model
        model = train_DeepQModel(age_bs.reshape(-1, 1), size_bs.reshape(-1, 1))  # independent variable: age_bs; dependent variable: size_bs
        
        # Predict
        age_bs_tensor = torch.tensor(age_bs, dtype=torch.float32).reshape(-1, 1)
        preds = model(age_bs_tensor)
        
        age_range_tensor = torch.tensor(age_range, dtype=torch.float32).reshape(-1, 1)
        y_test = model(age_range_tensor)
        
        for i, q in enumerate(quantiles):
            pred = preds[i].detach().numpy()
            pred = np.squeeze(pred)
            pred_boot[q].append(y_test[i].detach().numpy())

            # Store predictions for coverage calculation
            if abs(q - 0.1) < 0.001:
                q10_pred = pred
            elif abs(q - 0.9) < 0.001:
                q90_pred = pred
            
            # Calculate errors
            error_metrics[q]['mse'].append(mean_squared_error(size_bs, pred))
            error_metrics[q]['mae'].append(mean_absolute_error(size_bs, pred))
            error_metrics[q]['mape'].append(mean_absolute_percentage_error(size_bs, pred))

        # Calculate coverage proportion
        if q10_pred is not None and q90_pred is not None:
            q10_pred = np.squeeze(q10_pred)
            q90_pred = np.squeeze(q90_pred)
            coverage = calculate_coverage_proportion(size_bs, q10_pred, q90_pred)
            coverage_proportions.append(coverage)
    
    return pred_boot, error_metrics, coverage_proportions


    
def bootstrap_Xp50(x, y, n_bootstrap=2000):
    """Generic Bootstrap function to calculate Xp50 values
    
    Args:
        x: Predictor variable (age or weight)
        y: Binary response variable (maturity status)
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Array of bootstrap-estimated Xp50 values
    """
    values = []
    for _ in range(n_bootstrap):
        # Bootstrap resampling with replacement
        bs_indices = np.random.choice(len(x), len(x), replace=True)
        x_bs, y_bs = x.iloc[bs_indices], y.iloc[bs_indices]
        
        # Train-test split (20% test)
        x_train, _, y_train, _ = train_test_split(x_bs, y_bs, test_size=0.2)
        
        # Fit logistic regression
        model = LogisticRegression()  # penalty='none'
        model.fit(x_train.values.reshape(-1, 1), y_train)
        
        # alculate Xp50 (age/weight at 50% maturity probability)
        Xp50 = -model.intercept_[0] / model.coef_[0][0]
        values.append(Xp50)
    
    return np.array(values)