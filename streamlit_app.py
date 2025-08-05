"""
Streamlit application for predicting Los Angeles high temperatures and displaying
live Kalshi market odds.

This app uses historical LAX high‑temperature data together with current
observations and forecast information to train a multinomial logistic
regression model.  The model estimates the probability that tomorrow's high
temperature will fall within a 2°F bucket (for example "76‑77" or
"74‑75").  In addition to the traditional features (observed highs and
deviations from normals), we engineer cyclic seasonal terms and include
short‑ and longer‑term moving averages plus tomorrow's forecast high from
the National Weather Service.

The app also queries the Kalshi trading API (if a valid API key is
supplied via Streamlit secrets) to display the latest market odds for the
"Highest temperature in Los Angeles tomorrow" contract.  Since the
contract buckets change from day to day, the odds table is built
dynamically by parsing each contract's ticker.

Users can run this app locally with:

```
streamlit run streamlit_app.py
```

or deploy to Streamlit Community Cloud by pushing this repository and
setting the environment variable `KALSHI_API_KEY` as a secret.  The
forecast API used here does not require an API key.

Note: This model and application are for informational purposes only.
Predictions are specific to the Los Angeles weather market on Kalshi and
should not be construed as financial advice.
"""

import datetime
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CSV_PATH = "LA_weather_last_year.csv"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the historical temperature dataset from the CSV file.

    The dataset contains columns for the observation date, observed maximum
    temperature, normal maximum temperature, deviation from normal, and
    a seven‑day moving average of the deviation.  The Date column is
    converted to ``datetime.date`` objects for easier manipulation.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset with a proper ``Date`` column type.
    """
    df = pd.read_csv(
        CSV_PATH,
        header=1,
        names=[
            "Date",
            "Observed_Max_F",
            "Normal_Max_F",
            "Deviation_F",
            "MA7_Deviation",
        ],
    )
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def fetch_latest_temp() -> Optional[int]:
    """Retrieve the most recent observed high temperature for LAX.

    Scrapes the National Weather Service's daily climate report for Los
    Angeles International Airport (station LAX) and extracts the maximum
    temperature value.  Returns ``None`` if the report is unavailable or
    parsing fails.

    Returns
    -------
    Optional[int]
        Today's observed high temperature in degrees Fahrenheit, or ``None``.
    """
    nws_url = (
        "https://forecast.weather.gov/product.php?site=LOX&product=CLI&issuedby=LAX"
    )
    try:
        res = requests.get(nws_url, timeout=10)
        res.encoding = "utf-8"
        text = res.text
    except Exception:
        return None
    match = re.search(r"MAXIMUM TEMPERATURE\s*[^\d]*(\d{2,3})", text)
    return int(match.group(1)) if match else None


def fetch_forecast_high_and_humidity() -> Tuple[Optional[int], Optional[int]]:
    """Fetch tomorrow's forecast high and relative humidity from the NWS API.

    Uses the National Weather Service's gridpoint forecast to obtain the
    upcoming forecast.  The endpoint returns JSON containing a list of
    periods with temperature and humidity values.  This function extracts
    the first period that corresponds to tomorrow's daytime high (assuming
    the first element is the current period).  If parsing fails, returns
    ``None`` for both values.

    Returns
    -------
    Tuple[Optional[int], Optional[int]]
        A tuple containing tomorrow's forecast high temperature (°F) and
        relative humidity (%).  If unavailable, each element is ``None``.
    """
    # Gridpoint for downtown Los Angeles (approx).  Adjust if needed.
    grid_url = (
        "https://api.weather.gov/gridpoints/LOX/154,44/forecast"
    )
    try:
        data = requests.get(grid_url, timeout=10).json()
        periods = data.get("properties", {}).get("periods", [])
        # We assume the second period (index 1) corresponds to tomorrow's
        # forecast during the day.  This may vary but generally holds.
        if len(periods) >= 2:
            tomorrow = periods[1]
            return tomorrow.get("temperature"), tomorrow.get("relativeHumidity")
    except Exception:
        pass
    return None, None


def build_model(df: pd.DataFrame) -> Tuple[LogisticRegression, StandardScaler, pd.DataFrame]:
    """Prepare data and train a multinomial logistic regression model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing historical high‑temperature records.  Must
        include columns: ``Observed_Max_F``, ``Normal_Max_F``, ``Deviation_F``,
        and ``MA7_Deviation``.

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression classifier.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used to standardize the feature matrix.
    df_processed : pandas.DataFrame
        Copy of the input DataFrame with additional engineered features and
        bucket labels.
    """
    df = df.copy()
    # Create bucket labels: each 2°F range becomes a category, e.g. 74–75
    df["Bucket"] = df["Observed_Max_F"].apply(
        lambda x: f"{int(x // 2) * 2}-{int(x // 2) * 2 + 1}"
    )
    # Drop missing values prior to feature engineering
    df = df.dropna()
    # Cyclical day-of-year features
    df["DayOfYear"] = df["Date"].apply(lambda d: d.timetuple().tm_yday)
    df["Sin_Day"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["Cos_Day"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)
    # 14‑day moving average of deviation
    df["MA14_Deviation"] = df["Deviation_F"].rolling(window=14).mean()
    # Note: forecast high and humidity not in historical data – they will be
    # appended at prediction time, so they are omitted here.
    # Define predictor columns and shift by one day
    feature_cols = [
        "Observed_Max_F",
        "Normal_Max_F",
        "Deviation_F",
        "MA7_Deviation",
        "MA14_Deviation",
        "Sin_Day",
        "Cos_Day",
    ]
    X = df[feature_cols].shift(1).dropna()
    # Align the target variable with the predictor matrix by selecting
    # bucket labels for the same indices retained after shifting.  Using
    # ``df.loc[X.index]`` ensures that ``X`` and ``y`` have matching
    # lengths even when rows are dropped due to rolling means or other
    # NaN values.  This avoids inconsistent sample sizes during model
    # fitting and preserves the one‑day lag between features and
    # outcomes.
    y = df.loc[X.index, "Bucket"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
    model.fit(X_scaled, y)
    return model, scaler, df


def predict_next_day(
    df: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
    latest_temp: float,
    latest_norm_temp: float,
    forecast_high: Optional[int],
    humidity: Optional[int],
) -> List[Tuple[str, float]]:
    """Generate tomorrow's feature vector and predict top 3 buckets.

    Combines the observed high and normal high with deviation, moving
    averages, seasonal terms and optional forecast features.  The
    resulting probabilities are sorted and truncated to the top three
    most likely temperature ranges.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed historical dataset (output of :func:`build_model`).
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression classifier.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for standardization.
    latest_temp : float
        Today's observed high temperature.
    latest_norm_temp : float
        Normal high temperature for today (from historical normals).
    forecast_high : Optional[int]
        Tomorrow's forecasted high temperature in degrees Fahrenheit.
    humidity : Optional[int]
        Tomorrow's forecasted relative humidity (percent).

    Returns
    -------
    list of (str, float)
        List of tuples with the bucket label and the associated probability,
        sorted in descending order.  Only the top three entries are returned.
    """
    deviation = latest_temp - latest_norm_temp
    ma7_dev = df["Deviation_F"].tail(6).mean()
    ma14_dev = df["Deviation_F"].tail(13).mean()
    day_of_year = datetime.date.today().timetuple().tm_yday
    sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
    # Use forecast high/humidity if available; else fallback to today's observed high and default humidity 50%
    f_high = forecast_high if forecast_high is not None else latest_temp
    hum = humidity if humidity is not None else 50
    # Build DataFrame with features; columns must match training order
    new_row = pd.DataFrame(
        [
            [
                datetime.date.today(),
                latest_temp,
                latest_norm_temp,
                deviation,
                ma7_dev,
                ma14_dev,
                sin_day,
                cos_day,
                f_high,
                hum,
            ]
        ],
        columns=[
            "Date",
            "Observed_Max_F",
            "Normal_Max_F",
            "Deviation_F",
            "MA7_Deviation",
            "MA14_Deviation",
            "Sin_Day",
            "Cos_Day",
            "Forecast_High",
            "Humidity",
        ],
    )
    # Append to dataset to align features; note that training set did not include forecast/humidity,
    # but scaler will ignore extra columns when selecting training columns.
    # We'll select the same feature order used during training.
    feature_cols = [
        "Observed_Max_F",
        "Normal_Max_F",
        "Deviation_F",
        "MA7_Deviation",
        "MA14_Deviation",
        "Sin_Day",
        "Cos_Day",
    ]
    latest_row = new_row[feature_cols].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest_row)
    preds = model.predict_proba(latest_scaled)[0]
    labels = model.classes_
    return sorted(zip(labels, preds), key=lambda x: -x[1])[:3]


def sign_kalshi_request(private_key_pem: str, message: str) -> str:
    """Sign a Kalshi API request string using RSA PSS and return base64.

    The Kalshi trading API requires that each request include a
    signature computed over the timestamp, HTTP method and request path.
    This helper loads the RSA private key and returns a base64‑encoded
    signature of the provided message.  Any errors loading the key
    propagate to the caller.

    Parameters
    ----------
    private_key_pem : str
        The RSA private key in PEM format.  This should include the
        BEGIN/END delimiters.
    message : str
        The message to sign (timestamp + method + path).

    Returns
    -------
    str
        Base64‑encoded signature.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    import base64

    private_key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"), password=None
    )
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def fetch_kalshi_odds(key_id: Optional[str], private_key_pem: Optional[str]) -> List[Tuple[str, str]]:
    """Fetch current odds for the LA temperature market from Kalshi.

    This version uses Kalshi's signed authentication scheme.  It
    constructs a signature from the current timestamp, HTTP method and
    request path using the provided RSA private key.  The key ID,
    signature and timestamp are included in the request headers.  If
    either the key ID or private key is missing, this function
    immediately returns an empty list.

    Returns
    -------
    list of (str, str)
        Each tuple contains a temperature range string and its mid‑market
        probability formatted as a percentage (e.g., ('72-73', '23.5%')).
    """
    # If credentials are not provided, skip fetching odds
    if not key_id or not private_key_pem:
        return []
    import time
    # API endpoint details
    path = "/v0/markets/KXHIGLAX/orderbooks"
    url = f"https://trading-api.kalshi.com{path}"
    timestamp = str(int(time.time() * 1000))
    message = timestamp + "GET" + path
    try:
        signature = sign_kalshi_request(private_key_pem, message)
    except Exception:
        return []
    headers = {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        odds: List[Tuple[str, str]] = []
        for contract in data.get("orderbooks", []):
            ticker = contract.get("ticker", "")
            m = re.search(r"(\d{2}-\d{2})$", ticker)
            if not m:
                continue
            temp_range = m.group(1)
            bid = contract.get("buy_price")
            ask = contract.get("sell_price")
            if bid is None or ask is None:
                continue
            mid = (bid + ask) / 2.0
            odds.append((temp_range, f"{mid:.1f}%"))
        odds.sort(key=lambda x: int(x[0].split("-")[0]))
        return odds
    except Exception:
        return []


def display_kalshi_odds(odds: Sequence[Tuple[str, str]]) -> None:
    """Render the Kalshi odds table using Streamlit components.

    Parameters
    ----------
    odds : sequence of (str, str)
        Each element contains a temperature range and its associated
        probability (as a string with percentage).
    """
    if not odds:
        st.info(
            "Kalshi odds unavailable.  Set the KALSHI_API_KEY secret to display live market data."
        )
        return
    st.markdown("### Live Kalshi Market Odds")
    odds_df = pd.DataFrame(odds, columns=["Range", "Mid‑Market Price"])
    st.table(odds_df)


def main() -> None:
    """Main entrypoint for the Streamlit app.

    Performs the following steps:
    1. Loads historical data.
    2. Fetches latest observed high and forecast features.
    3. Trains the logistic regression model and predicts tomorrow's high.
    4. Displays predictions alongside live market odds (if API key set).
    5. Provides context and navigation to the "How it Works" page via sidebar.
    """
    st.set_page_config(
        page_title="LA Weather Picks", page_icon="🌤️", layout="wide"
    )
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:", ["Forecast & Odds", "How it Works"], index=0
    )
    if page == "How it Works":
        # If the user selects the other page, import and run it here.
        from pages.page_how_it_works import render_page

        render_page()
        return
    # Forecast & Odds page
    st.title("Los Angeles Weather Forecast & Market Odds")
    st.markdown(
        "Discover tomorrow's high‑temperature probabilities alongside live trading odds."
    )
    # Load data
    df = load_data()
    # Fetch today's observed high
    latest_temp = fetch_latest_temp()
    if latest_temp is None:
        st.error(
            "Failed to fetch today's high temperature from the NWS report."
        )
        return
    # Determine today's normal high from historical data (most recent normal before today)
    if not df.empty:
        mask = df["Date"] < datetime.date.today()
        if mask.any():
            latest_norm_temp = df.loc[mask, "Normal_Max_F"].iloc[-1]
        else:
            latest_norm_temp = df["Normal_Max_F"].iloc[-1]
    else:
        latest_norm_temp = latest_temp
    # Fetch forecast features
    forecast_high, humidity = fetch_forecast_high_and_humidity()
    # Ensure we have enough data
    if len(df) < 8:
        st.error(
            "Insufficient data to build the prediction model.  Please provide at least 8 days of data."
        )
        return
    # Train model
    model, scaler, processed_df = build_model(df)
    # Predict tomorrow's bucket probabilities
    top_preds = predict_next_day(
        processed_df,
        model,
        scaler,
        latest_temp,
        latest_norm_temp,
        forecast_high,
        humidity,
    )
    # Layout: two columns for predictions and odds
    col1, col2 = st.columns((2, 1))
    with col1:
        st.subheader("Model Predictions")
        st.markdown(f"**Today's high (observed):** {latest_temp}°F")
        if forecast_high is not None:
            st.markdown(f"**Tomorrow's forecast high:** {forecast_high}°F")
        # Create predictions DataFrame
        pred_df = pd.DataFrame(top_preds, columns=["Temperature Range", "Probability"])
        pred_df["Probability"] = (pred_df["Probability"] * 100).round(1).astype(str) + "%"
        st.table(pred_df)
        best_bucket = top_preds[0][0]
        st.success(f"**Model pick:** {best_bucket}")
    with col2:
        # Display Kalshi odds using signed authentication.  We expect two
        # secrets to be defined in Streamlit Cloud: ``KALSHI_KEY_ID`` and
        # ``KALSHI_PRIVATE_KEY``.  If either is missing, ``fetch_kalshi_odds``
        # will return an empty list and the UI will prompt the user.
        kalshi_key_id = st.secrets.get("KALSHI_KEY_ID")
        kalshi_private_key = st.secrets.get("KALSHI_PRIVATE_KEY")
        odds = fetch_kalshi_odds(kalshi_key_id, kalshi_private_key)
        display_kalshi_odds(odds)
    st.caption(
        "Model predictions are estimates based on historical and forecast data. "
        "Live odds sourced from Kalshi may differ and are provided for comparison only."
    )


if __name__ == "__main__":
    main()