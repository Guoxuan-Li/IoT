import pandas as pd
import requests
import time
import os
import numpy as np
from pytrends.request import TrendReq
from datetime import datetime, timedelta

# Define save path
dir_path = os.path.dirname(os.path.abspath(__file__))
folder = 'data'

# time range
start_time = "2026-01-26"
end_time = "2026-02-07"

def p(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def directories():
    data_dir = os.path.join(dir_path, folder)
    for d in ['raw', 'processed']:
        path = os.path.join(data_dir, d)
        os.makedirs(path, exist_ok=True)
    return data_dir

data_root = directories()

# Collect physical data (Open-Meteo)
def collect_physical_weather():
    p(f"Start downloading real weather data")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 51.5074, "longitude": -0.1278, # London's Location
        "start_date": start_time, "end_date": end_time,
        "hourly": ["temperature_2m", "precipitation", "rain", "wind_speed_10m"],
        "timezone": "Europe/London"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        # Check whether the data is abnormal
        resp.raise_for_status()
        df = pd.DataFrame(resp.json()['hourly'])
        
        # Save the raw physical data
        raw_path = os.path.join(data_root, 'raw', 'Weather_Physical_Raw.csv')
        df.to_csv(raw_path, index=False)
        
        # Data clean and save
        df['time'] = pd.to_datetime(df['time'])
        df.rename(columns={'time': 'date'}, inplace=True)
        df.set_index('date', inplace=True)
        
        processed_path = os.path.join(data_root, 'processed', 'Weather_Physical_Final.csv')
        df.to_csv(processed_path)
        p(f"Final physical data saved to: {processed_path}")

    except Exception as e:
        p(f"Download physical data failed: {e}")

# Collect social data (Google Trends)
def collect_social_hourly(keyword):
    p(f"Start collecting keyword: '{keyword}' ")

    # first week
    tf_week1 = "2026-01-26T00 2026-02-02T00"
    # second week
    tf_week2 = "2026-02-01T00 2026-02-07T23"
    
    # Initial pytrends
    pytrends = TrendReq(hl='en-GB', tz=0, timeout=(10,25))
    
    try:
        # Downloading week1's data (2026-01-26 to 02-02)
        p(f"Downloading week1's data (Hourly)")
        pytrends.build_payload([keyword], cat=0, timeframe=tf_week1, geo='GB-ENG')
        df_week1 = pytrends.interest_over_time()
        # save raw data
        df_week1.to_csv(os.path.join(data_root, 'raw', f'{keyword}_week1_Raw.csv'))
        time.sleep(1)

        # Downloading week2's data (2026-02-01 to 02-8)
        p(f"Downloading week2's data (Hourly)")
        pytrends.build_payload([keyword], cat=0, timeframe=tf_week2, geo='GB-ENG')
        df_week2 = pytrends.interest_over_time()
        # save raw data
        df_week2.to_csv(os.path.join(data_root, 'raw', f'{keyword}_week2_Raw.csv'))

        if df_week1.empty or df_week2.empty:
            p(f"'{keyword}' download failed. No data available.")
            return

        # Data processing
        overlap = df_week1.index.intersection(df_week2.index)
        
        if len(overlap) > 0:
            p("Start Processing data")
            val_week1 = df_week1.loc[overlap, keyword].mean()
            val_week2 = df_week2.loc[overlap, keyword].mean()
            
            # calculate the ratio
            if val_week2 != 0:
                ratio = val_week1 / val_week2
                df_week2[keyword] = df_week2[keyword] * ratio
            
            final_df = pd.concat([df_week1, df_week2])
            final_df = final_df[~final_df.index.duplicated(keep='first')]
        else:
            final_df = pd.concat([df_week1, df_week2])
        
        # normalization
        mx = final_df[keyword].max()
        if mx > 0: 
            final_df[keyword] = (final_df[keyword] / mx) * 100
        
        full_range = pd.date_range(start=final_df.index.min(), end=final_df.index.max(), freq='h')
        final_df = final_df.reindex(full_range)
        final_df[keyword] = final_df[keyword].interpolate(method='linear')
        final_df = final_df[[keyword]].round(2)
        final_df.index.name = 'date'

        # save files
        save_path = os.path.join(data_root, 'processed', f'{keyword}_final.csv')
        final_df.to_csv(save_path)
        p(f"'{keyword}' processed file saved to: {save_path}")
        
    except Exception as e:
        p(f"Google Trends Data downloaded fail: {e}")

# Main Loop
if __name__ == "__main__":
    p("Start collecting IoT data")
    directories()
    
    # collect physical data
    collect_physical_weather()
    
    # collect social data (Traffic)
    collect_social_hourly("Traffic")
    
    # collect social data (Weather)
    collect_social_hourly("Weather")
    
    p("ALL DONE")