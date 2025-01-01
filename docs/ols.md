# Ordinary-Least-Squares(OLS)
## 使い方
```python
import pyregress
import pandas as pd

df = pd.read_csv('path/to/your/file') #データセットの読み込み

model = pyregress.OLS(
    stock = df['stock'].to_numpy(),
    dxy = df['dxy'].to_numpy(),
    nasdaq = df['nasdaq'].to_numpy()
)

model.model_option.dep_var_name = 'stock' #被説明変数の指定
model.fitting_option.method = 'qr' #回帰係数計算方法の指定
model.fitting_option.cov_type = 'HC1' #共分散行列計算方法の指定

result = model.fit() #resultはデータクラス
summary = result.summary() #summaryは辞書オブジェクト

#続きを記述する...
```

## 詳細
### モデルの初期化
`pyregress.OLS()`は初期化時に、キーワード引数のみ（`np.ndarray`）を受け取ります。  
`(変数名) = (np.ndarrayオブジェクト)`となるように引数を渡してください。  
>[!WARNING]  
>全ての引数の`shape`が同じでない場合、例外が送出されます。

### オプションの指定
`model.model_option`には、以下の項目が格納されています。  
- `dep_var_name`: 被説明変数の変数名。デフォルトは`y`  
　　
<br/><br/>

`model.fitting_option`には、以下の項目が格納されています。  
- `method`: 回帰係数の計算方法。以下の計算方法をサポートしています。
    - `pinv` (デフォルト)
    - `qr`

<br/>

- `cov_type`: 共分散行列の計算方法。以下の計算方法をサポートしています。
    - `nonrobust` (デフォルト)
    - `fixed scale`
    - `HC0`
    - `HC1`
    - `HC2`
    - `HC3`
    - `HAC`
    - `hac-panel`
    - `hac-groupsum`
    - `cluster`


### 分析の実行
`model.fit()`を実行すると、分析処理が開始されます。  
このメソッドの戻り値は`OLS_Regression_Result`オブジェクトです。このオブジェクトは、以下のプロパティ及びメソッドを持っています。
- プロパティ
    - `params`
    - `residuals`
    - `fitted_values`
    - `n_obs`
    - `n_params`
    - `sum_of_squared`
    - `mean_squared_error`
    - `covariance_matrix`
    - `std_errors`
- メソッド
    - `summary`

### 分析結果をまとめる
先述の`OLS_Regression_Result`オブジェクトの`summary`メソッドを実行すると、辞書オブジェクトが返却されます。この辞書は以下のような形式になっています。  
`columns`以下の構造は、`変数名: (辞書)`となっています。
```python
{
    "header": {
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
    },
    "columns": {
        "x1": {
            "coef": float,
            "std-err": float,
            "z": float,
            "P>|z|": float,
            "0.025>": float,
            "<0.975": float
        },
        #変数の数に応じて続く...
    },
    "footer": {
        "Omnibus": float,
        "Prob(Omnibus)": float,
        "Skew": float,
        "Kurtosis": float,
        "Durbin-Watson": float,
        "Jarque-Bera(JB)": float,
        "Prob(JB)": float,
        "Cond. No.": float  
    },
    "notes": str
}
```