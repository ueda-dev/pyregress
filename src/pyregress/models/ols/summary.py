from typing import *

Header = TypedDict('Header', {
    "Dep. Variable:": str,
    "Model": str,
    "Method": str,
    "Date": str,
    "Time": str,
    "No. Observations": int,
    "Df Residuals": int,
    "Df Model": int,
    "Covariance Type": str,
    "R-squared": float,
    "Adj. R-squared": float,
    "F-statistc": float,
    "Prob(F-statistic)": float,
    "Log-Likelihood": float,
    "AIC": float,
    "BIC": float
})

Column = TypedDict('Column', {
    "coef": float,
    "std-err": float,
    "z": float,
    "P>|z|": float,
    "0.025>": float,
    "<0.975": float
})

Footer = TypedDict('Footer', {
    "Omnibus": float,
    "Prob(Omnibus)": float,
    "Skew": float,
    "Kurtosis": float,
    "Durbin-Watson": float,
    "Jarque-Bera(JB)": float,
    "Prob(JB)": float,
    "Cond. No.": float
})

class OLS_Summary(TypedDict):
    header: Header
    columns: Dict[str, Column]
    footer: Footer
    notes: str