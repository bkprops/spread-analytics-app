# Online Spread Analytics App

This directory hosts the Streamlit analytics dashboard. The app loads the Google Sheet directly, which keeps it compatible with Streamlit Community Cloud (no backend service required).

## Data source

The dashboard downloads the dataset from the Google Sheet URL defined by `ONLINE_SPREAD_SHEET_URL` (defaults to the shared sheet at `https://docs.google.com/spreadsheets/d/1ijAq9nDO4XK0xnQ0HRBzQp8HLXzijPe3NodvrSDcF5c`). Override the variable if you maintain a separate copy of the sheet; an empty value is treated as a configuration error.

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. When configuring the app on Streamlit Community Cloud, set the “Main file” to `app/frontend/streamlit_app.py`.
3. The root `requirements.txt` (or `pyproject.toml`) supplies the necessary dependencies. Add `ONLINE_SPREAD_SHEET_URL` under “Advanced settings” if you need a different sheet.

## Run Streamlit locally

From the repository root:

```bash
streamlit run app/frontend/streamlit_app.py
```

Set `ONLINE_SPREAD_SHEET_URL=<your-url>` beforehand if you want to override the default sheet.
