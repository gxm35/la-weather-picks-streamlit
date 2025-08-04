import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import re
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CSV_PATH = "LA_weather_last_year.csv"

def fetch_latest_temp():
    nws_url = "https://forecast.weather.gov/product.php?site=LOX&product=CLI&issuedby=LAX"
    res = requests.get(nws_url)
    res.encoding = 'utf-8'
    text = res.text
    match = re.search(r"MAXIMUM TEMPERATURE\s*[^\d]*(\d{2,3})", text)
    if match:
        return int(match.group(1))
    else:
        return None

def load_data():
    df = pd.read_csv(CSV_PATH, header=1, names=['Date','Observed_Max_F','Normal_Max_F','Deviation_F','MA7_Deviation'])
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def build_model(df):
    df['Bucket'] = df['Observed_Max_F'].apply(lambda x: f"{int(x//2)*2}–{int(x//2)*2+1}")
    df = df.dropna()
    X = df[['Observed_Max_F','Normal_Max_F','Deviation_F','MA7_Deviation']].shift(1).dropna()
    y = df['Bucket'].iloc[1:]
    X, y = X.iloc[1:], y.iloc[1:]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
    model.fit(X_scaled, y)
    return model, scaler, df

def predict_next_day(df, model, scaler, latest_temp, latest_norm_temp):
    deviation = latest_temp - latest_norm_temp
    ma7_dev = df['Deviation_F'].tail(6).mean()
    new_row = pd.DataFrame([[datetime.date.today(), latest_temp, latest_norm_temp, deviation, ma7_dev]],
                           columns=['Date','Observed_Max_F','Normal_Max_F','Deviation_F','MA7_Deviation'])
    df_appended = pd.concat([df, new_row], ignore_index=True)
    latest_row = df_appended.iloc[-1][['Observed_Max_F','Normal_Max_F','Deviation_F','MA7_Deviation']].values.reshape(1,-1)
    latest_scaled = scaler.transform(latest_row)
    preds = model.predict_proba(latest_scaled)[0]
    labels = model.classes_
    return sorted(zip(labels, preds), key=lambda x: -x[1])[:3]

def main():
    st.title('LA Weather Picks')
    st.subheader('Logistic regression prediction for LA high temperature')
    df = load_data()
    latest_temp = fetch_latest_temp()
    if latest_temp is None:
        st.error('Could not fetch latest temperature from NWS.')
        return
    if len(df) > 0:
        latest_norm_temp = df[df['Date'] < datetime.date.today()]['Normal_Max_F'].iloc[-1]
    else:
        latest_norm_temp = latest_temp
    model, scaler, df_processed = build_model(df)
    top_preds = predict_next_day(df_processed, model, scaler, latest_temp, latest_norm_temp)
    st.write(f"**Today's high:** {latest_temp} °F")
    st.write("### Tomorrow's most likely temperature buckets:")
    for bucket, prob in top_preds:
        st.write(f"**{bucket}°F:** {prob*100:.1f}%")
    st.write(f"**My pick:** {top_preds[0][0]}°F bucket")
    st.caption('This model is for informational purposes and specific to the Los Angeles weather market on Kalshi.')

if __name__ == '__main__':
    main()
