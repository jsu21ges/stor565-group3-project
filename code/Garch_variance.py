import pandas as pd
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor, DMatrix, train
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
import random
from xgboost import plot_importance
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import t
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))  # Mean Absolute Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100  # Avoid division by zero
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100
    r2 = r2_score(y_true, y_pred)
    mda = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    nrmse = rmse / np.std(y_true)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'SMAPE': smape, 'R2': r2, 'MDA': mda, 'NRMSE': nrmse}

btc = pd.read_csv("/Users/maxnemecek/Downloads/BTC.csv")
rates = pd.read_csv("/Users/maxnemecek/Downloads/DFF.csv")
VIX = pd.read_csv("/Users/maxnemecek/Downloads/VIXCLS.csv")
InfExp = pd.read_csv("/Users/maxnemecek/Downloads/T5YIE.csv")

btc["date"] = pd.to_datetime(btc["date"])
rates["date"] = pd.to_datetime(rates["observation_date"])
VIX["date"] = pd.to_datetime(VIX["observation_date"])
InfExp["date"] = pd.to_datetime(InfExp["observation_date"])

rates = rates.drop('observation_date', axis=1)
VIX = VIX.drop('observation_date', axis=1)
InfExp = InfExp.drop('observation_date', axis=1)

start_date = "2015-01-01"
end_date = "2025-01-01"

btc = btc[(btc["date"] >= start_date) & (btc["date"] <= end_date)]
rates = rates[(rates["date"] >= start_date) & (rates["date"] <= end_date)]
VIX = VIX[(VIX["date"] >= start_date) & (VIX["date"] <= end_date)]
InfExp = InfExp[(InfExp["date"] >= start_date) & (InfExp["date"] <= end_date)]

df = pd.merge(btc, rates, on='date', how='inner')
df = pd.merge(df, VIX, on='date', how='inner')
df = pd.merge(df, InfExp, on='date', how='inner')

df['DFF']=df['DFF'].bfill()
df['VIXCLS']=df['VIXCLS'].bfill()
df['T5YIE']=df['T5YIE'].bfill()
##df['returns'] = 100 * np.log(df['close'] / df['close'].shift(1))
df['returns'] = 100*((df['close']-df['close'].shift(1))/df['close'].shift(1))
df = df.dropna()

df['Lag1_Return'] = df['returns'].shift(1)
df['Lag2_Return'] = df['returns'].shift(2)
df['VIXCLS_1'] = df['VIXCLS'].shift(1)
df['DFF_1'] = df['DFF'].shift(1)
df['T5YIE_1'] = df['T5YIE'].shift(1)
df['MA7'] = df['close'].shift(1).rolling(window=7).mean()
df['Abs_Lag1_Return'] = df['returns'].shift(1).abs()
df['Rolling_Vol'] = df['returns'].shift(1).rolling(window=5).std()
df['Rolling_Max_Return'] = df['returns'].shift(1).rolling(window=5).max()
df = df.dropna()

train_size = int(0.8 * len(df))
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

features = ['VIXCLS']
target = 'returns'

# Prepare train and test data
X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

garch_model = arch_model(
    y_train,
    x=X_train,
    mean='ARX', 
    lags=1,
    vol='GARCH',
    p=1, 
    q=1,
    dist='t'
)

garch_fit = garch_model.fit(disp='off')
print(garch_fit.summary())

nu = garch_fit.params.get('nu', None)
print(f"\nEstimated nu (degrees of freedom): {nu:.4f}")
if nu:
    print(f" - nu={nu:.4f} indicates {'heavy' if nu < 6 else 'moderate' if nu < 10 else 'thin'} tails")
    print(f" - Lower nu means more extreme events are modeled, suitable for Bitcoin returns")
residuals = garch_fit.resid / garch_fit.conditional_volatility
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=50, density=True, alpha=0.7, label='Standardized Residuals')
x = np.linspace(-5, 5, 100)
plt.plot(x, t.pdf(x, df=nu), label=f't-dist (nu={nu:.2f})')
plt.xlabel('Standardized Residuals')
plt.ylabel('Density')
plt.title('Standardized Residuals vs. t-Distribution')
plt.legend()
plt.show()

mean_forecasts = []
variance_forecasts = []

for i in range(len(X_test)):
    # Combine training data and test data up to t-1
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    if i == 0:
        current_y = y_train
        current_x = X_train
    else:
        current_y = np.concatenate([y_train, y_test[:i]])
        current_x = np.vstack([X_train, X_test[:i]])

    model = arch_model(
        current_y,
        x=current_x,
        mean='ARX',
        lags=1,
        vol='GARCH',
        p=1,
        q=1,
        dist='t'
    )
    fit = model.fit(disp='off')
    try:
        x_i = np.array(X_test.iloc[i])
        if x_i.shape != (len(features),):
            raise ValueError(f"X_test[{i}] shape {x_i.shape} != ({len(features)},)")
        if np.any(np.isnan(x_i)):
            raise ValueError(f"NaN in X_test[{i}]")
        x_forecast = x_i[:, np.newaxis, np.newaxis]
    except Exception as e:
        print(f"x_forecast prep failed at i={i}: {e}")
        mean_forecasts.append(np.nan)
        variance_forecasts.append(np.nan)
        continue
    
    # Forecast one step ahead
    try:
        forecast = fit.forecast(horizon=1, x=x_forecast, reindex=False)
        mean_forecasts.append(forecast.mean.values[-1, -1])
        variance_forecasts.append(forecast.variance.values[-1, -1])
    except Exception as e:
        print(f"Forecast failed at i={i}: {e}")
        mean_forecasts.append(np.nan)
        variance_forecasts.append(np.nan)

# Convert forecasts to numpy arrays
mean_forecasts = np.array(mean_forecasts)
variance_forecasts = np.array(variance_forecasts)

# Add forecasts to df (align with test set)
df_test['garch_mean'] = np.nan
df_test['garch_variance'] = np.nan
df_test.loc[:, 'garch_mean'] = mean_forecasts
df_test.loc[:, 'garch_variance'] = variance_forecasts

df_train.loc[:, 'garch_mean'] = df_train['returns']-garch_fit.resid
df_train.loc[:, 'garch_variance'] = garch_fit.conditional_volatility

df = pd.concat([df_train, df_test])
df = df.dropna()

plt.figure(figsize=(12, 6))
plt.plot(df_test['date'], np.sqrt(variance_forecasts), label='GARCH Volatility Forecast', alpha=0.7)
plt.plot(df_test['date'], np.abs(y_test), label='Absolute Returns (Proxy for Volatility)', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('GARCH(1,1) Volatility Forecast vs Actual')
plt.legend()
plt.show()
"""
rolling_vol = y_test.rolling(window=5).apply(lambda x: np.std(x)**2).fillna(method='bfill')
variance_metrics_rolling = compute_metrics(rolling_vol, variance_forecasts)
print("\nGARCH(1,1) Variance Metrics (vs. Rolling Volatility):")
for metric, value in variance_metrics_rolling.items():
    print(f"{metric}: {value:.4f}")
"""

window_size = 7
df['fwd_rolling_vol'] = df['returns'].shift(-window_size).rolling(window=window_size).std()
df = df.dropna()
df['Realized Volatility'] = np.nan
df['Realized Volatility'] = np.abs(df['returns']).copy()
fwd_vol_mean = df['fwd_rolling_vol'].mean()
fwd_vol_std = df['fwd_rolling_vol'].std()
df['fwd_rolling_vol_norm'] = (df['fwd_rolling_vol'] - fwd_vol_mean) / fwd_vol_std

train_size = int(0.8 * len(df))
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

features = ['VIXCLS','MA7', 'Rolling_Max_Return', 'Abs_Lag1_Return', 'garch_variance', 'Rolling_Vol']
target = 'Realized Volatility'

X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

garch_metrics = compute_metrics(df['Realized Volatility'], df['garch_variance'])

print("\nGARCH(1,1) Variance Forecast Metrics:")
for metric, value in garch_metrics.items():
    print(f"{metric}: {value:.4f}")
# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print("Final data check...")
print("NaNs in X_train_scaled:", np.isnan(X_train_scaled).sum())
print("NaNs in y_train:", y_train.isna().sum())
print("Infinities in y_train:", np.isinf(y_train).sum())
print("NaNs in X_test_scaled:", np.isnan(X_test_scaled).sum())
print("NaNs in y_test:", y_test.isna().sum())
print("Infinities in y_test:", np.isinf(y_test).sum())

params = {
    'eta': 0.01,
    'max_depth': 5,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': 42
    }
dtrain = DMatrix(X_train_scaled, label=y_train)
dtest = DMatrix(X_test_scaled, label=y_test)
evals = [(dtrain, 'train'), (dtest, 'test')]
    
xgb_model = train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=True)
    
    # Wrap the trained model in XGBRegressor for scikit-learn compatibility
xgb_regressor = XGBRegressor()
xgb_regressor._Booster = xgb_model  # Assign the trained booster
xgb_model = xgb_regressor

# Predict on test set
y_pred = xgb_model.predict(X_test_scaled)


# Compute metrics
xgb_metrics = compute_metrics(y_test, y_pred)

# Print metrics
print("\nXGBoost Metrics for Volatility Prediction:")
for metric, value in xgb_metrics.items():
    print(f"{metric}: {value:.4f}")
    
feature_importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)

# Get gain-based importance
gain_importances = xgb_model.get_booster().get_score(importance_type='gain')
gain_importances = [gain_importances.get(f'f{i}', 0) for i in range(len(features))]
importance_df['Importance (Gain)'] = gain_importances / np.sum(gain_importances)  # Normalize

importance_df = importance_df.sort_values(by='Importance (Gain)', ascending=False)
print("\nFeature Importance:")
print(importance_df)

# Plot 1: Using xgboost.plot_importance (Weight-based)
plt.figure(figsize=(10, 6))
plot_importance(xgb_model, importance_type='weight', title='XGBoost Feature Importance (Weight)')
plt.show()

# Plot 2: Custom seaborn bar plot (Gain-based)
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance (Gain)', 
    y='Feature', 
    data=importance_df,
    palette='viridis'
)
plt.title('XGBoost Feature Importance (Gain)')
plt.xlabel('Normalized Importance (Gain)')
plt.ylabel('Feature')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_test['date'].iloc[:len(y_test)], y_test, 
         label='True Volatility', alpha=0.7)
plt.plot(df_test['date'].iloc[:len(y_pred)], y_pred, 
         label='XGBoost Predicted Volatility', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('XGBoost Predictions vs. True Volatility')
plt.legend()
plt.show()

rf_model = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)

plt.figure(figsize=(12, 6))
plt.plot(df_test['date'].iloc[:len(y_test)], y_test, 
         label='True Volatility', alpha=0.7)
plt.plot(df_test['date'].iloc[:len(y_pred)], y_pred, 
         label='RF Predicted Volatility', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('RF Predictions vs. True Volatility')
plt.legend()
plt.show()

importance_df = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})
print(importance_df.sort_values(by='Importance', ascending=False))

rf_metrics = compute_metrics(y_test, y_pred)

# Print metrics
print("\nRandom Forest Metrics for Volatility Prediction:")
for metric, value in rf_metrics.items():
    print(f"{metric}: {value:.4f}")

abs_returns_mean = np.abs(df_train['returns']).mean()
print(f"\nMean of Absolute Returns (Training): {abs_returns_mean:.6f}")

y_pred_constant = np.full_like(y_test, abs_returns_mean)
constant_metrics = compute_metrics(y_test, y_pred_constant)
