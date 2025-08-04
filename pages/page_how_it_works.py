"""
Streamlit page to explain how the LA Weather Picks model operates.

This page is imported lazily from ``streamlit_app.py`` when users
navigate to "How it Works" via the sidebar.  It provides a detailed
explanation of the model’s inputs, feature engineering steps,
logistic regression training, and integration with live market data.
"""

import streamlit as st


def render_page() -> None:
    """Render the "How it Works" page for the Streamlit app."""
    st.title("How the LA Weather Picks Model Works")
    st.markdown(
        """
        ### Overview
        The LA Weather Picks application uses a multinomial logistic regression model to
        forecast which 2°F temperature range tomorrow’s high will fall into.  The model
        is trained on a full year of historical high temperatures from Los Angeles
        International Airport (LAX).  For each day in the dataset we compute several
        features including:

        • **Deviation from normal:** Difference between the observed maximum and the
          climatological normal maximum for that date.  Positive values indicate
          warmer‑than‑normal days and negative values cooler‑than‑normal days.
        • **Moving averages:** Seven‑ and fourteen‑day rolling averages of the deviation
          capture short‑ and medium‑term trends in how temperatures are evolving.
        • **Seasonal cycles:** Sinusoidal terms (`sin(2π day/365.25)` and `cos(2π day/365.25)`) model
          the annual seasonal cycle, allowing the regression to anticipate warmer
          summer months and cooler winter months.
        • **Forecast features:** The model optionally ingests tomorrow’s forecast high
          and relative humidity from the National Weather Service.  These features
          incorporate the latest meteorological guidance into the probability
          estimates.

        These features are standardized (mean‑centered and scaled to unit variance) and
        fed into a multinomial logistic regression.  The regression estimates
        probabilities across multiple mutually exclusive temperature buckets (e.g.
        68‑69, 70‑71, etc.).  Training uses a one‑day lag so that the features
        available at the time of prediction correspond to previous observations.

        ### Integrating Live Market Data
        When an API key is provided, the app retrieves current odds from Kalshi's
        **Highest temperature in Los Angeles tomorrow** market via their trading API.
        For each available contract, the app extracts the bucket label from the
        ticker symbol (e.g., “72-73”) and computes the mid‑market price between
        the best bid and ask.  These odds are displayed alongside the model’s
        predictions so users can compare statistical forecasts with market
        sentiment.

        ### Disclaimers
        The LA Weather Picks app is for informational and educational purposes only.
        It does not provide financial advice.  Trading in prediction markets
        involves risk, and users should conduct their own due diligence.
        """
    )
