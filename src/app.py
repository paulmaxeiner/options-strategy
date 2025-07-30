import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import streamlit as st
import pytz as pytz

K=240
r=0.02
sigma=0.5

def black_scholes_call(S,T):
    if T == 0:
        return max(S - K, 0)  # Handle expiration edge case
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


stock_data = {
    'T': [0.005,0.004,0.003,0.002,0.001,0],
    'S': [241.05,241.06,241.03,241.02,241.07,241.09]
}

df=pd.DataFrame(stock_data)

st.line_chart(df)

df['call_price'] = df.apply(lambda row:black_scholes_call(row['S'],row['T']), axis=1)

st.write(df)




dt1_str = "2025-07-15 20:30:00+00:00"
est = pytz.timezone('US/Eastern')
utc = pytz.utc
fmt = '%Y-%m-%d %H:%M:%S %Z%z'
dt1 = datetime.fromisoformat(dt1_str)
st.write(dt1.astimezone(est).strftime(fmt))

def to_EST(dt):
    return dt.datetime.astimezone(est)




# Define the second datetime object (4pm on the same date as dt1)
# We need to explicitly set the time and timezone.
dt2 = dt1.replace(hour=17, minute=0, second=0, microsecond=0)

# Calculate the difference
time_difference = dt2 - dt1
st.write(time_difference.total_seconds())

data = yf.download('SPY', start='2025-07-15', end='2025-07-16', interval='1m')
data = data.reset_index()
st.write(data['Datetime'])


data['Date'] = data['Datetime'].dt.strftime(fmt)



st.write(data['Date'])


st.write(data['Close'])
data['hi']=data['Close'].pct_change()

from datetime import datetime, time, timedelta

def calculate_minutes_to_4pm_est(input_datetime):
    """
    Calculates the number of minutes passed from 4 PM EST on the same day
    as the input_datetime.

    Args:
        input_datetime (datetime): A datetime object representing the current time.
                                   It can be in any timezone, but it will be
                                   converted to EST for calculation.

    Returns:
        int: The number of minutes passed from 4 PM EST. Returns 0 if the
             input_datetime is at or before 4 PM EST on the same day.
             Returns -1 if the input_datetime is on a different day than 4 PM EST.
    """
    est = pytz.timezone('America/New_York')

    # Convert input_datetime to EST
    if input_datetime.tzinfo is None:
        # Assume input_datetime is in local time and convert to EST
        local_tz = datetime.now().astimezone().tzinfo
        input_datetime_est = local_datetime.astimezone(est)
    else:
        input_datetime_est = input_datetime.astimezone(est)

    # Define 4 PM EST on the same day as the input
    four_pm_est = est.localize(datetime.combine(input_datetime_est.date(), time(16, 0, 0)))

    # Check if the input_datetime is on the same day as 4 PM EST
    if input_datetime_est.date() != four_pm_est.date():
        return -1 # Indicates different days

    # Calculate the difference if input_datetime is after 4 PM EST
    if input_datetime_est < four_pm_est:
        time_difference = four_pm_est - input_datetime_est
        minutes_passed = int(time_difference.total_seconds() / 60)
        return minutes_passed
    else:
        return 0 # Input is at or before 4 PM EST



def minutes_from_4pm(datetime_series):
    """
    Converts a Pandas Series of datetime objects to the number of minutes
    from 4 PM on the same day.

    Args:
        datetime_series (pd.Series): A Pandas Series containing datetime objects.

    Returns:
        pd.Series: A Series containing the number of minutes from 4 PM.
                   Positive values indicate time after 4 PM, negative values
                   indicate time before 4 PM.
    """
    four_pm = time(16, 0, 0)  # Represents 4 PM (16:00:00)
    est = pytz.timezone('America/New_York')
    four_pm = est.localize(datetime.combine(datetime_series[0].date(), time(16, 0, 0)))
    
    

    
    # Create a datetime object for 4 PM on the same day as each entry
    # This involves combining the date from the series with the fixed 4 PM time
    four_pm_datetimes = datetime_series.dt.normalize() + pd.to_timedelta(four_pm.hour, unit='h')
    
    # Calculate the time difference (timedelta)
    time_difference = datetime_series - four_pm_datetimes
    
    # Convert timedelta to minutes
    minutes = time_difference.dt.total_seconds() / 60 /60/24/365
    
    return minutes


data['tim'] = data.apply(lambda row: minutes_from_4pm(row['Datetime']), axis=1)
st.write(data)
