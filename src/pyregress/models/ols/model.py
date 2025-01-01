from __future__ import annotations
from typing import *
from types import NoneType
import numpy as np
from dataclasses import dataclass
from pyregress.models.ols.summary import OLS_Summary, Header, Column, Footer
from datetime import datetime
from scipy.stats import f as f_dist
import math

@dataclass
class OLS_Model_Option:
    dep_var_name: str = 'y'

@dataclass
class OLS_Fitting_Option:
    method: Literal['pinv', 'qr'] = "pinv"
    cov_type: Literal['nonrobust', 'fixed scale', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC', 'hac-panel', 'hac-groupsum', 'cluster'] = 'nonrobust'
    cov_kwds: Any | None = None

@dataclass
class OLS_Regression_Result:
    _parent: OLS
    params: np.typing.NDArray[np.float64]
    residuals: np.typing.NDArray
    fitted_values: np.typing.NDArray
    n_obs: int
    n_params: int
    sum_of_squared: float
    mean_squared_error: float
    covariance_matrix: np.ndarray
    std_errors: np.ndarray

    def summary(self) -> OLS_Summary:
        # Get variable names
        var_names = [
            key for key in self._parent._terms.keys() if key != self._parent.model_option.dep_var_name
        ] + ['const']  # Add 'const' explicitly if necessary

        # Calculate R-squared and Adjusted R-squared
        total_sum_of_squares = np.sum((self.fitted_values - np.mean(self.fitted_values)) ** 2)
        r_squared = 1 - self.sum_of_squared / total_sum_of_squares
        adj_r_squared = 1 - (1 - r_squared) * (self.n_obs - 1) / (self.n_obs - self.n_params - 1)

        # Calculate F-statistic (simplified)
        explained_variance = total_sum_of_squares / (self.n_params - 1)
        residual_variance = self.sum_of_squared / (self.n_obs - self.n_params)
        f_statistic = explained_variance / residual_variance
        prob_f_statistic = 1 - self._f_cdf(f_statistic, self.n_params - 1, self.n_obs - self.n_params)

        # Header
        header: Header = {
            "Dep. Variable:": self._parent.model_option.dep_var_name,
            "Model": "OLS",
            "Method": "Least Squares",
            "Date": datetime.now().strftime("%a, %d %b %Y"),
            "Time": datetime.now().strftime("%H:%M:%S"),
            "No. Observations": self.n_obs,
            "Df Residuals": self.n_obs - self.n_params,
            "Df Model": self.n_params - 1,
            "Covariance Type": self._parent.fitting_option.cov_type,
            "R-squared": r_squared,
            "Adj. R-squared": adj_r_squared,
            "F-statistc": f_statistic,
            "Prob(F-statistic)": prob_f_statistic,
            "Log-Likelihood": -self.n_obs / 2 * np.log(2 * np.pi * self.mean_squared_error) - self.sum_of_squared / (2 * self.mean_squared_error),
            "AIC": 2 * self.n_params - 2 * (-self.n_obs / 2 * np.log(2 * np.pi * self.mean_squared_error) - self.sum_of_squared / (2 * self.mean_squared_error)),
            "BIC": self.n_params * np.log(self.n_obs) - 2 * (-self.n_obs / 2 * np.log(2 * np.pi * self.mean_squared_error) - self.sum_of_squared / (2 * self.mean_squared_error))
        }

        # Columns
        columns: Dict[str, Column] = {}
        for var_name, coef, se in zip(var_names, self.params, self.std_errors):
            z_value = coef / se
            p_value = 2 * (1 - self._z_score_cdf(abs(z_value)))  # Two-tailed p-value
            conf_int_low = coef - 1.96 * se
            conf_int_high = coef + 1.96 * se
            columns[var_name] = {
                "coef": coef,
                "std-err": se,
                "z": z_value,
                "P>|z|": p_value,
                "0.025>": conf_int_low,
                "<0.975": conf_int_high
            }

        # Footer
        footer: Footer = {
            "Omnibus": 35.923,  # Placeholder, replace with actual calculation if needed
            "Prob(Omnibus)": 0.283,  # Placeholder
            "Skew": 0.698,  # Placeholder
            "Kurtosis": 5.391,  # Placeholder
            "Durbin-Watson": 2.0,  # Placeholder, replace with actual calculation
            "Jarque-Bera(JB)": 79.517,  # Placeholder
            "Prob(JB)": 5.41e-18,  # Placeholder
            "Cond. No.": 3.97e+05  # Placeholder
        }

        # Notes
        notes = '' #後で実装する。基準値を調査中。

        return {
            "header": header,
            "columns": columns,
            "footer": footer,
            "notes": notes
        }

    def _z_score_cdf(self, z: float) -> float:
        """Calculate the cumulative distribution function for a z-score."""
        return (1 + math.erf(z / np.sqrt(2))) / 2

    def _f_cdf(self, f: float, dfn: int, dfd: int) -> float:
        """Calculate the cumulative distribution function for an F-statistic."""
        return f_dist.cdf(f, dfn, dfd)

class OLS:
    def __init__(self, **terms: np.ndarray) -> None:
        self.model_option = OLS_Model_Option()
        self.fitting_option = OLS_Fitting_Option()

        #checking shape. if not same, then raise exception
        def check_shape():
            value = None
            for arr in terms.values():
                if type(value) == NoneType:
                    value = arr
                else:
                    if not value.shape == arr.shape:
                        raise ValueError('all array must have same length')
        check_shape()

        #set terms
        self._terms = terms

        #if not const in terms, then add it
        if not terms.get('const'):
            self._terms['const'] = np.ones(terms.get(list(terms.keys())[0]).shape[0])

    def fit(self):
        def _calculate_covariance_matrix(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
            n, k = X.shape
            cov_type = self.fitting_option.cov_type

            if cov_type == 'nonrobust':
                # Classic OLS covariance matrix
                sigma2 = np.sum(residuals ** 2) / (n - k)
                XtX_inv = np.linalg.inv(X.T @ X)
                return sigma2 * XtX_inv

            elif cov_type in {'HC0', 'HC1', 'HC2', 'HC3'}:
                # Heteroscedasticity-robust covariance matrix
                XtX_inv = np.linalg.inv(X.T @ X)
                diag_residuals = residuals ** 2

                if cov_type == 'HC0':
                    omega = np.diag(diag_residuals)
                elif cov_type == 'HC1':
                    omega = np.diag(diag_residuals * (n / (n - k)))
                elif cov_type == 'HC2':
                    leverage = np.diag(X @ XtX_inv @ X.T)
                    omega = np.diag(diag_residuals / (1 - leverage))
                elif cov_type == 'HC3':
                    leverage = np.diag(X @ XtX_inv @ X.T)
                    omega = np.diag(diag_residuals / (1 - leverage) ** 2)

                return XtX_inv @ X.T @ omega @ X @ XtX_inv

            elif cov_type == 'fixed scale':
                # Example: fixed scale (you can define your own scaling logic if needed)
                scale = self.fitting_option.cov_kwds.get("scale", 1.0)
                sigma2 = scale ** 2
                XtX_inv = np.linalg.inv(X.T @ X)
                return sigma2 * XtX_inv

            else:
                raise ValueError(f"Unsupported covariance type: {cov_type}")

        def _pinv(y:np.ndarray, X:np.ndarray):
            XtX_inv = np.linalg.pinv(X.T @ X)
            XtY = X.T @ y
            return XtX_inv @ XtY
        
        def _qr(y:np.ndarray, X:np.ndarray):
            Q, R = np.linalg.qr(X)
            return np.linalg.solve(R, Q.T @ y)
        
        def _get_result(params: np.ndarray, y:np.ndarray, X:np.ndarray):
            residuals = y - X @ params
            fitted_values = X @ params
            n_obs = len(y)
            n_params = len(params)
            ssr = np.sum(residuals ** 2)  # Sum of squared residuals
            mse = ssr / (n_obs - n_params)  # Mean squared error
            cov_matrix = _calculate_covariance_matrix(X, residuals)
            std_errors = np.sqrt(np.diag(cov_matrix))

            return OLS_Regression_Result(self, params, residuals, fitted_values, n_obs, n_params, ssr, mse, cov_matrix, std_errors)

        dep_var = self._terms.get(self.model_option.dep_var_name)
        exp_vars = np.stack([value for key, value in self._terms.items() if not key == self.model_option.dep_var_name], axis=1)

        if not dep_var:
            raise ValueError(f'dep. var "{self.model_option.dep_var_name}" does not exist. please specify OLS.model_option.dep_var_name')

        if self.fitting_option.method == 'pinv':
            params = _pinv(dep_var, exp_vars)
        elif self.fitting_option.method == 'qr':
            params = _qr(dep_var, exp_vars)
        else:
            raise ValueError(f'method: {self.fitting_option.method} is not supported')
        
        return _get_result(params, dep_var, exp_vars)
