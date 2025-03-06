import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Optional: This helps ensure matplotlib doesn't try to open in an interactive backend
import matplotlib
matplotlib.use("Agg")

# Try importing Prophet
try:
    from prophet import Prophet
    has_prophet = True
except ImportError:
    has_prophet = False

# Try importing statsmodels for Exponential Smoothing
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    has_es = True
except ImportError:
    has_es = False


###############################################################################
# 1. Data Loading & Utilities
###############################################################################
CSV_URL = "https://deluxe555.pythonanywhere.com/files/gas_data.csv"

@st.cache_data
def load_csv_data(csv_url=CSV_URL):
    """
    Loads the CSV from a URL, parses Timestamp as datetime, sorts by time, and returns the raw DataFrame.
    Cached with @st.cache_data to avoid re-loading on every rerun unless the URL changes (implicitly if data at URL changes).
    **Warning: SSL certificate verification is attempted but might fail due to your pandas version or system configuration.
    Please see the error details for more information.**
    """
    try:
        df = pd.read_csv(csv_url) # Removed verify=False
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df.dropna(subset=["Timestamp"], inplace=True)
        df.sort_values("Timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data from URL '{csv_url}': {e}")
        st.error(f"Detailed error: {e}") # Display detailed error for debugging
        st.warning("SSL certificate verification may have failed, or your pandas version is outdated. Please check the detailed error above. If SSL issues persist, consider updating your pandas library or addressing system certificate issues.")
        return pd.DataFrame()

def remove_outliers_iqr(df, column="FastGasPrice", factor=1.5):
    """Single-step outlier removal using IQR."""
    if df.empty or column not in df.columns:
        return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)].copy()

def aggregate_5min(df):
    """Resample to 5-minute intervals on key columns."""
    if df.empty or "Timestamp" not in df.columns:
        return df
    df = df.set_index("Timestamp").sort_index()
    agg_dict = {
        "FastGasPrice": "mean",
        "SafeGasPrice": "mean",
        "ProposeGasPrice": "mean",
        "SuggestBaseFee": "mean"
    }
    df_agg = df.resample("5T").agg(agg_dict).dropna().reset_index()
    return df_agg

def feature_engineering(df):
    """Add hour/day features, lag/rolling columns, drop NaNs."""
    if df.empty or "Timestamp" not in df.columns:
        return df
    df = df.copy()
    df.sort_values("Timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["SafeGasPrice_lag1"] = df["SafeGasPrice"].shift(1)
    df["FastGasPrice_lag1"] = df["FastGasPrice"].shift(1)
    df["SafeGasPrice_roll5"] = df["SafeGasPrice"].rolling(5).mean()
    df["FastGasPrice_roll5"] = df["FastGasPrice"].rolling(5).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


###############################################################################
# 2. Models
###############################################################################
def naive_forecast(df):
    """Return the last known FastGasPrice (Naive approach)."""
    if df.empty or "FastGasPrice" not in df.columns:
        return np.nan
    return df["FastGasPrice"].iloc[-1]


def train_prophet_model(df, cp_scale=0.05):
    """
    Train Prophet on (Timestamp, FastGasPrice).
    If Prophet is not installed or insufficient data, return None.
    """
    if not has_prophet or len(df) < 2 or "FastGasPrice" not in df.columns:
        return None

    prophet_df = df[["Timestamp", "FastGasPrice"]].rename(columns={
        "Timestamp": "ds",
        "FastGasPrice": "y"
    })
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")

    model = Prophet(
        changepoint_prior_scale=cp_scale,
        seasonality_mode="additive",
        daily_seasonality=True
    )
    try:
        model.fit(prophet_df)
        return model
    except Exception as e:
        st.warning(f"Prophet training error: {e}")
        return None


def forecast_prophet_steps(model, steps=12):
    """Forecast 'steps' future intervals (5-min freq)."""
    if not model:
        return pd.DataFrame()
    try:
        future = model.make_future_dataframe(periods=steps, freq="5T")
        forecast_df = model.predict(future)
        return forecast_df
    except Exception as e:
        st.warning(f"Prophet forecast error: {e}")
        return pd.DataFrame()


def es_forecast_steps(df, steps=1):
    """Exponential Smoothing forecast for 'steps' intervals."""
    if (not has_es) or len(df) < 2 or "FastGasPrice" not in df.columns:
        return np.nan
    series = df.sort_values("Timestamp")["FastGasPrice"].copy()
    series.index = range(len(series))
    try:
        model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
        fc = model.forecast(steps)
        return fc.iloc[-1] if len(fc) > 0 else np.nan
    except Exception as e:
        st.warning(f"Exponential Smoothing error: {e}")
        return np.nan


###############################################################################
# 3. Rolling Backtest
###############################################################################
def rolling_backtest_single_step(
    df,
    window=50,
    cp_scale=0.05,
    models=("Naive", "Prophet", "ES")
):
    """
    Single-step rolling forecast for each window:
      - Evaluate absolute error & directional accuracy.
    Returns a dict with average errors and average directional accuracy.
    """
    if df.empty or "FastGasPrice" not in df.columns:
        return {}
    n = len(df)
    if n <= window:
        st.warning(f"Not enough data (need > {window} rows).")
        return {}

    errors = {m: [] for m in models}
    directional = {m: [] for m in models}

    for start_idx in range(n - window):
        train_df = df.iloc[start_idx : start_idx + window].copy()
        test_idx = start_idx + window
        actual_val = df["FastGasPrice"].iloc[test_idx]
        last_train_val = train_df["FastGasPrice"].iloc[-1]

        # Naive
        if "Naive" in models:
            naive_pred = last_train_val
            errors["Naive"].append(abs(actual_val - naive_pred))
            dir_actual = np.sign(actual_val - last_train_val)
            dir_naive = np.sign(naive_pred - last_train_val)
            directional["Naive"].append(1 if dir_actual == dir_naive else 0)

        # Prophet
        if "Prophet" in models:
            prop_model = train_prophet_model(train_df, cp_scale=cp_scale)
            if prop_model:
                fc_df = forecast_prophet_steps(prop_model, steps=1)
                if not fc_df.empty:
                    yhat = fc_df["yhat"].iloc[-1]
                    errors["Prophet"].append(abs(actual_val - yhat))
                    dir_prophet = np.sign(yhat - last_train_val)
                    dir_actual = np.sign(actual_val - last_train_val)
                    directional["Prophet"].append(1 if dir_actual == dir_prophet else 0)
                else:
                    errors["Prophet"].append(np.nan)
                    directional["Prophet"].append(0)
            else:
                errors["Prophet"].append(np.nan)
                directional["Prophet"].append(0)

        # Exponential Smoothing
        if "ES" in models:
            es_pred = es_forecast_steps(train_df, steps=1)
            if pd.isna(es_pred):
                errors["ES"].append(np.nan)
                directional["ES"].append(0)
            else:
                errors["ES"].append(abs(actual_val - es_pred))
                dir_es = np.sign(es_pred - last_train_val)
                dir_actual = np.sign(actual_val - last_train_val)
                directional["ES"].append(1 if dir_actual == dir_es else 0)

    # Compute average (mean) errors/directional accuracy
    avg_errors = {m: np.nanmean(errors[m]) for m in models}
    avg_directional = {m: np.nanmean(directional[m]) for m in models}

    return {
        "errors": avg_errors,
        "directional": avg_directional
    }


###############################################################################
# 4. Hyperparameter Search for Prophet
###############################################################################
def find_best_prophet_cp_scale(df, window=50, scale_candidates=[0.01, 0.05, 0.1, 0.3]):
    """
    Example function to pick the best cp_scale from a given list
    by running a rolling backtest and comparing average error.
    Returns (best_scale, best_error) or (None, None) if fail.
    """
    best_scale = None
    best_error = float("inf")

    for cp in scale_candidates:
        st.write(f"Testing changepoint_prior_scale={cp}...")
        results = rolling_backtest_single_step(
            df,
            window=window,
            cp_scale=cp,
            models=("Prophet",)  # only test Prophet here
        )
        if "errors" in results and "Prophet" in results["errors"]:
            err = results["errors"]["Prophet"]
            st.write(f"  -> Average error: {err}")
            if err < best_error:
                best_error = err
                best_scale = cp

    if best_scale is not None:
        st.success(f"Best cp_scale={best_scale} with avg error={best_error}")
        return best_scale, best_error
    else:
        st.warning("Could not find a best scale; returning defaults.")
        return None, None


###############################################################################
# 5. Multi-Horizon Forecast
###############################################################################
def multi_horizon_forecast(df, steps_list=[1,3,12], cp_scale=0.05):
    """
    Generates multi-step forecasts (5-min intervals) using Naive, Prophet, and ES.
    steps_list = [1,3,12] -> 5 min, 15 min, 60 min horizons
    """
    # Naive
    naive_val = naive_forecast(df)

    # Prophet
    prop_model = train_prophet_model(df, cp_scale=cp_scale)
    max_steps = max(steps_list)
    prop_fc = forecast_prophet_steps(prop_model, steps=max_steps) if prop_model else pd.DataFrame()

    prophet_vals = {}
    if not prop_fc.empty:
        tail_fc = prop_fc.tail(max_steps).reset_index(drop=True)
        for s in steps_list:
            idx = s - 1
            if idx >= 0 and idx < len(tail_fc):
                prophet_vals[s] = tail_fc["yhat"].iloc[idx]
            else:
                prophet_vals[s] = np.nan
    else:
        prophet_vals = {s: np.nan for s in steps_list}

    # ES
    es_vals = {}
    for s in steps_list:
        es_vals[s] = es_forecast_steps(df, steps=s)

    # Build final forecast table
    rows = []
    for s in steps_list:
        row = {
            "Horizon (x5min)": s,
            "Naive": naive_val,
            "Prophet": prophet_vals[s],
            "ExponentialSmoothing": es_vals[s]
        }
        rows.append(row)
    return pd.DataFrame(rows)


###############################################################################
# 6. Main Streamlit App
###############################################################################
def main():
    st.title("Ethereum Gas Forecasting Dashboard (CSV from URL, Enhanced Backtest)")

    # Reload the CSV from URL if user clicks
    if st.button("Refresh Data from URL"):
        st.experimental_rerun()

    # Load the data from URL
    df_raw = load_csv_data() # Loads from CSV_URL by default
    if df_raw.empty:
        st.stop()

    # Outlier removal (optional)
    if st.checkbox("Remove outliers on FastGasPrice? (IQR=1.5)", value=True):
        df_clean = remove_outliers_iqr(df_raw, "FastGasPrice", factor=1.5)
    else:
        df_clean = df_raw.copy()

    # Aggregate & feature engineering
    df_agg = aggregate_5min(df_clean)
    df_feat = feature_engineering(df_agg)

    if len(df_feat) < 10:
        st.warning("Not enough data after cleaning, aggregation, and feature engineering.")
        st.stop()

    st.subheader("Data Preview After Preprocessing")
    st.dataframe(df_feat.head(10))

    # Quick EDA - Time Series Plot
    st.subheader("Time Series Plot (FastGasPrice)")
    fig, ax = plt.subplots()
    ax.plot(df_feat["Timestamp"], df_feat["FastGasPrice"], label="FastGasPrice")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Gwei")
    ax.set_title("5-Min Aggregated FastGasPrice")
    ax.legend()
    st.pyplot(fig)

    # Hyperparameter Search (Optional)
    st.header("Prophet Hyperparameter Selection (Optional)")
    st.markdown("Try multiple changepoint_prior_scale values and pick the best via rolling backtest.")

    candidate_scales = st.multiselect(
        "Candidate cp_scale values",
        [0.01, 0.05, 0.1, 0.3, 0.5],
        default=[0.01, 0.05, 0.1]
    )
    window_size = st.number_input("Rolling Window Size", 10, 500, 50, step=5)

    if st.button("Run Hyperparameter Search"):
        best_cp, best_err = find_best_prophet_cp_scale(df_feat, window=window_size, scale_candidates=candidate_scales)
        if best_cp:
            st.write(f"Use cp_scale={best_cp} for final backtests/forecasts.")
        else:
            st.write("No best cp_scale found; you can try again or pick your own.")

    # Enhanced Rolling Backtest
    st.header("Rolling Backtest (Single-Step Forecast)")
    st.markdown("Evaluate Naive, Prophet, and ES for single-step rolling predictions.")
    user_cp_scale = st.number_input("Choose a cp_scale for Prophet", 0.001, 1.0, 0.05, step=0.01)
    if st.button("Run Rolling Backtest"):
        results = rolling_backtest_single_step(
            df_feat,
            window=window_size,
            cp_scale=user_cp_scale,
            models=("Naive", "Prophet", "ES")
        )
        if results:
            st.subheader("Average Errors")
            st.write(results["errors"])
            st.subheader("Directional Accuracy")
            st.write(results["directional"])
        else:
            st.warning("No backtest results were returned.")

    # Multi-Horizon Forecast
    st.header("Multi-Horizon Forecast")
    st.markdown("(Ex: 1 step = 5 min, 3 steps = 15 min, 12 steps = 60 min, etc.)")
    if st.button("Generate Multi-Horizon Forecast"):
        forecast_df = multi_horizon_forecast(df_feat, steps_list=[1, 3, 12], cp_scale=user_cp_scale)
        st.dataframe(forecast_df)

    st.success("Done! This app now loads data from a URL and uses a more robust rolling backtest + optional Prophet hyperparameter search.")


if __name__ == "__main__":
    main()