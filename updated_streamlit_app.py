import pandas as pd
import numpy as np
import datetime
import requests
import re
from bs4 import BeautifulSoup
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CSV_PATH = "LA_weather_last_year.csv"


def fetch_latest_temp():
    """
    Retrieve the most recent high temperature for Los Angeles Airport (LAX)
    from the National Weather Service (NWS) climate report.  If the report
    cannot be parsed, this function returns None.
    """
    nws_url = "https://forecast.weather.gov/product.php?site=LOX&product=CLI&issuedby=LAX"
    try:
        res = requests.get(nws_url)
        res.encoding = 'utf-8'
        text = res.text
    except Exception:
        return None
    match = re.search(r"MAXIMUM TEMPERATURE\s*[^\d]*(\d{2,3})", text)
    return int(match.group(1)) if match else None


def load_data():
    """
    Load the historical temperature dataset from the CSV file.  The dataset
    contains columns for the observation date, observed maximum temperature,
    normal maximum temperature, deviation from normal, and a seven‑day moving
    average of the deviation.  The Date column is converted to Python
    datetime.date objects for easier time manipulation.
    """
    df = pd.read_csv(
        CSV_PATH,
        header=1,
        names=[
            'Date',
            'Observed_Max_F',
            'Normal_Max_F',
            'Deviation_F',
            'MA7_Deviation',
        ],
    )
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df


def build_model(df: pd.DataFrame):
    """
    Given a DataFrame of historical temperatures, compute bucket labels,
    engineer additional seasonal features, standardize predictors and fit a
    multinomial logistic regression.  Returns the fitted model, the scaler
    used for normalization, and the processed DataFrame.
    """
    # Create labels representing 2‑degree temperature ranges.  For example,
    # an observed high of 76°F falls into the bucket "76-77".
    df = df.copy()
    df['Bucket'] = df['Observed_Max_F'].apply(
        lambda x: f"{int(x // 2) * 2}-{int(x // 2) * 2 + 1}"
    )

    # Drop rows with missing values before feature engineering.
    df = df.dropna()

    # Add cyclical features to capture annual seasonality.  DayOfYear runs
    # from 1 to 365/366.  Sin_Day and Cos_Day encode this angle on the
    # unit circle.
    df['DayOfYear'] = df['Date'].apply(lambda d: d.timetuple().tm_yday)
    df['Sin_Day'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25)
    df['Cos_Day'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25)

    # Define the set of predictors.  We use a one‑day lag for all
    # continuous predictors because tomorrow’s outcome depends on today’s
    # conditions.  The bucket labels (target) are correspondingly shifted.
    feature_cols = [
        'Observed_Max_F',
        'Normal_Max_F',
        'Deviation_F',
        'MA7_Deviation',
        'Sin_Day',
        'Cos_Day',
    ]
    X = df[feature_cols].shift(1).dropna()
    y = df['Bucket'].iloc[1:]
    # Align X and y after shifting
    X, y = X.iloc[1:], y.iloc[1:]

    # Standardize predictors and fit the logistic regression model.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=500
    )
    model.fit(X_scaled, y)
    return model, scaler, df


def predict_next_day(
    df: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
    latest_temp: float,
    latest_norm_temp: float,
):
    """
    Given the processed DataFrame, trained model and today’s observed and
    normal highs, compute the predictive features for tomorrow and return
    the top three most likely temperature buckets with their probabilities.
    """
    deviation = latest_temp - latest_norm_temp
    ma7_dev = df['Deviation_F'].tail(6).mean()
    day_of_year = datetime.date.today().timetuple().tm_yday
    sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
    cos_day = np.cos(2 * np.pi * day_of_year / 365.25)

    # Assemble the feature row for tomorrow.  Date is included for
    # completeness but excluded from the final feature matrix passed to
    # the model.
    new_row = pd.DataFrame(
        [
            [
                datetime.date.today(),
                latest_temp,
                latest_norm_temp,
                deviation,
                ma7_dev,
                sin_day,
                cos_day,
            ]
        ],
        columns=[
            'Date',
            'Observed_Max_F',
            'Normal_Max_F',
            'Deviation_F',
            'MA7_Deviation',
            'Sin_Day',
            'Cos_Day',
        ],
    )
    # Append to DataFrame and extract the latest feature vector.
    df_appended = pd.concat([df, new_row], ignore_index=True)
    latest_row = df_appended.iloc[[-1]][
        [
            'Observed_Max_F',
            'Normal_Max_F',
            'Deviation_F',
            'MA7_Deviation',
            'Sin_Day',
            'Cos_Day',
        ]
    ].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest_row)
    preds = model.predict_proba(latest_scaled)[0]
    labels = model.classes_
    # Sort descending by probability and return the top 3
    return sorted(zip(labels, preds), key=lambda x: -x[1])[:3]


def main():
    """
    Streamlit app entrypoint.  Loads data, trains the model and displays
    predictions along with explanatory text.  Provides a clean and
    responsive UI with headings, a table of predictions and a highlighted
    model pick.  Includes a disclaimer noting the application’s intended
    use for informational purposes within the LA weather market on Kalshi.
    """
    st.set_page_config(
        page_title='LA Weather Picks', page_icon='🌤️', layout='centered'
    )
    st.title('LA Weather Picks')
    st.markdown(
        '### Predict LA high temperatures using logistic regression with seasonal features'
    )
    st.markdown(
        'This tool fetches the latest high temperature from the National Weather Service and '
        'uses a logistic regression model trained on last year’s data—including deviation '
        'from normal and seasonal sinusoid features—to forecast tomorrow’s high‑temperature range.'
    )

    df = load_data()
    latest_temp = fetch_latest_temp()
    if latest_temp is None:
        st.error('Could not fetch the latest temperature from the NWS report.')
        return

    # Determine the most recent normal maximum temperature from the dataset.
    if len(df) > 0:
        # Use the last available normal max up to (but not including) today.
        mask = df['Date'] < datetime.date.today()
        if mask.any():
            latest_norm_temp = df.loc[mask, 'Normal_Max_F'].iloc[-1]
        else:
            latest_norm_temp = df['Normal_Max_F'].iloc[-1]
    else:
        latest_norm_temp = latest_temp

    # Ensure we have enough historical records to train the model.
    if len(df) < 8:
        st.error(
            'Insufficient data to build the prediction model. '
            'Please provide at least 8 days of data.'
        )
        return

    # Train the model and make predictions.
    model, scaler, df_processed = build_model(df)
    top_preds = predict_next_day(
        df_processed, model, scaler, latest_temp, latest_norm_temp
    )

    st.markdown(f"#### Today's high (observed): **{latest_temp}°F**")
    st.markdown('#### Top predicted temperature ranges for tomorrow:')

    # Prepare a DataFrame for display
    pred_df = pd.DataFrame(top_preds, columns=['Temperature range', 'Probability'])
    pred_df['Probability'] = (pred_df['Probability'] * 100).round(1).astype(str) + '%'
    st.table(pred_df)

    # Highlight the top prediction
    best_bucket = top_preds[0][0]
    st.success(f"**Model pick:** {best_bucket}")

    st.caption(
        'This model is for informational purposes only and is designed specifically '
        'for the Los Angeles weather market on Kalshi.'
    )


if __name__ == '__main__':
    main()
