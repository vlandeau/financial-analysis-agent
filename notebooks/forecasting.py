# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm

current_path = Path.cwd()
output_data_path = current_path / ".." / "data" / "output"
# %matplotlib inline

# %% [markdown]
# # Analyze revenue

# %%
revenue_per_quarter_df = pd.read_parquet(output_data_path / "revenue.parquet")

# %%
px.line(revenue_per_quarter_df)

# %%
px.line(revenue_per_quarter_df - revenue_per_quarter_df.shift(4))

# %%
px.line(revenue_per_quarter_df / revenue_per_quarter_df.shift(4))

# %%
px.line(np.log(revenue_per_quarter_df) - np.log(revenue_per_quarter_df.shift(4)))

# %% [markdown]
# # Forecast revenue

# %%
df_with_features = revenue_per_quarter_df.copy()
df_with_features["log_total_revenue"] = np.log(df_with_features["Total Revenue"])
df_with_features["time_trend"] = np.arange(len(df_with_features))
df_with_features["quarter"] = df_with_features.index.quarter
seasonal_dummies = pd.get_dummies(df_with_features["quarter"], prefix="Q", drop_first=True)
df_with_features = pd.concat([df_with_features, seasonal_dummies], axis=1)
df_with_features

# %% [markdown]
# ## Forecast revenue

# %%
features = ["time_trend", "Q_2", "Q_3", "Q_4"]

X = df_with_features[features]
y = df_with_features["Total Revenue"]

n_splits = 4
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)

predictions = []
actuals = []
fold_indices = []

model = LinearRegression()

for _, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)

    prediction_revenue = model.predict(X_test)[0]
    actual_revenue = y_test.iloc[0]

    predictions.append(prediction_revenue)
    actuals.append(actual_revenue)
    fold_indices.append(test_index[0]) # Store the index of the test point
    
    print(f"Fold {i+1}: Trained on {len(X_train)} quarters, Forecasted {prediction_revenue:.2f}, Actual was {actual_revenue:.2f}")

# %%
mape = mean_absolute_percentage_error(actuals, predictions)
print(f"Overall Cross-Validation MAPE: {mape:.2%}")

mape_last_three_quarters = mean_absolute_percentage_error(actuals[1:], predictions[1:])
print(f"Cross-Validation MAPE for last three quarters: {mape_last_three_quarters:.2%}")

# %%
results_df = pd.DataFrame({
    'Actual Revenue': actuals,
    'Forecasted Revenue': predictions
}, index=df_with_features.index[fold_indices])
results_df

# %%
px.line(results_df)

# %%
model = LinearRegression()
model.fit(df_with_features[features], df_with_features["Total Revenue"])
pd.DataFrame({
    "feature": features,
    "coef": model.coef_,
})

# %%
model.intercept_

# %% [markdown]
# ## Forecast log of revenue

# %%
features = ["time_trend", "Q_2", "Q_3", "Q_4"]

X = df_with_features[features]
y = df_with_features["log_total_revenue"]

n_splits = 4
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)

predictions = []
actuals = []
fold_indices = []

model = LinearRegression()

for _, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    
    log_forecast = model.predict(X_test)

    prediction_revenue = np.exp(log_forecast[0])
    actual_revenue = np.exp(y_test.iloc[0])

    predictions.append(prediction_revenue)
    actuals.append(actual_revenue)
    fold_indices.append(test_index[0]) # Store the index of the test point
    
    print(f"Fold {i+1}: Trained on {len(X_train)} quarters, Forecasted {prediction_revenue:.2f}, Actual was {actual_revenue:.2f}")

# %%
mape = mean_absolute_percentage_error(actuals, predictions)
print(f"Overall Cross-Validation MAPE: {mape:.2%}")

mape_last_three_quarters = mean_absolute_percentage_error(actuals[1:], predictions[1:])
print(f"Cross-Validation MAPE for last three quarters: {mape_last_three_quarters:.2%}")

# %%
results_df = pd.DataFrame({
    'Actual Revenue': actuals,
    'Forecasted Revenue': predictions
}, index=df_with_features.index[fold_indices])
results_df

# %%
px.line(results_df)

# %%
model = LinearRegression()
model.fit(df_with_features[features], df_with_features["log_total_revenue"])
pd.DataFrame({
    "feature": features,
    "coef": model.coef_,
})

# %%
model.intercept_

# %% [markdown]
# # Compare log linear regression to exponential smoothing

# %%
y = df_with_features["Total Revenue"]

ets = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="mul",
    seasonal_periods=4,
).fit(optimized=True)

ets_forecast = ets.forecast(4)
ets_forecast

# %%
ets = ExponentialSmoothing(
    y,
    trend="add",
    seasonal="mul",
    seasonal_periods=4,
).fit(optimized=True)

ets_forecast = ets.forecast(4)

# %%
df_with_ets_predictions = pd.concat([df_with_features, pd.DataFrame({"ets_forecast": ets_forecast})])
df_with_ets_predictions.drop(columns=["Q_2", "Q_3", "Q_4"], inplace=True)
df_with_ets_predictions["time_trend"] = np.arange(len(df_with_ets_predictions))
df_with_ets_predictions["quarter"] = df_with_ets_predictions.index.quarter
seasonal_dummies = pd.get_dummies(df_with_ets_predictions["quarter"], prefix="Q", drop_first=True)
df_with_ets_predictions = pd.concat([df_with_ets_predictions, seasonal_dummies], axis=1)
df_with_ets_predictions

# %%
X = df_with_features[features]
y = df_with_features["log_total_revenue"]

model = LinearRegression().fit(X, y)
df_with_ets_predictions["log_linear_forecast"] = np.exp(model.predict(df_with_ets_predictions[features]))

# %%
px.line(df_with_ets_predictions[["Total Revenue", "ets_forecast", "log_linear_forecast"]])

# %% [markdown]
# # Optimistic and pessimistic predictionsimport statsmodels.api as sm

# %%
X_train_full = df_with_features[features].astype(float)
X_train_full = sm.add_constant(X_train_full)
y_train_full = df_with_features["log_total_revenue"]

model_ols = sm.OLS(y_train_full, X_train_full).fit()

# %%
future_dates = pd.date_range(start='2024-03-31', periods=4, freq='Q')
X_future = pd.DataFrame(index=future_dates)
X_future['time_trend'] = np.arange(len(df_with_features), len(df_with_features) + 4)
X_future['quarter'] = X_future.index.quarter
future_seasonal_dummies = pd.get_dummies(X_future['quarter'], prefix='Q', drop_first=True)
X_future = pd.concat([X_future, future_seasonal_dummies], axis=1)
X_future = sm.add_constant(X_future, has_constant='add')
X_future = X_future.astype(float)
for col in X_train_full.columns:
    if col not in X_future.columns:
        X_future[col] = 0

# %%
X_future

# %%
pred_ols = model_ols.get_prediction(X_future[X_train_full.columns])
interval_ols_log = pred_ols.summary_frame(alpha=0.05)

interval_ols = np.exp(interval_ols_log)
interval_ols

# %%
df_with_prediction_range = pd.concat([df_with_features, interval_ols, df_with_ets_predictions[["ets_forecast"]]], axis=1)

# %%
px.line(df_with_prediction_range[["Total Revenue", "mean", "obs_ci_lower", "obs_ci_upper", "ets_forecast"]])

# %%
