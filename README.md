# Cyber-Physical-Social System based on London

An automated end-to-end platform designed to explore the correlations between physical environmental conditions (weather) and related social behavioural responses in London, UK.

## Project Overview
This project constructs a Cyber-Physical-Social System (CPSS) to analyze the relationship between urban environmental conditions and public behaviour. By integrating real-time meteorological data (Physical Sensing) with Google Trends search indices (Social Sensing), the platform identifies patterns, anomalies, and lagging effects in urban dynamics.

## System Architecture
The system adopts a pure software distributed architecture, balancing deployment cost with analytical depth.

1.  **Data Collection Layer:** Automated retrieval of 2-meter temperature, precipitation, rain, and 10-meter wind speed via Open-Meteo API. Keywords "Traffic" and "Weather" as social indices via Google Trends.
2.  **Network Layer:** Implements HTTPS session reuse, 15s timeout, and exponential backoff retry strategies for high reliability.
3.  **Storage Layer:** Saved raw data to /data/raw folder and processed data to /data/processed folder as CSV files.
4.  **Analytics Layer:** Time-series smoothing (Low-pass filter), Pearson Correlation, Cross-Correlation Function, Decomposition (STL), and Frequency analysis (FFT).
5.  **Visualization:** Interactive dashboard built with Streamlit and Plotly.

## Getting Started

### Prerequisites
* Python 3.10 or higher
* A stable internet connection (for API requests)
* Python libraries: streamlit, pandas, numpy, plotly, scipy, statsmodels, pytrends, requests, matplotlib

### Installation
Install libraries:
```
pip install -r requirements.txt
```

### Running the Platform
Launch the interactive dashboard locally:
```
streamlit run app.py
```
Change the file path (app.py) as your local path

## Methodology

### Data Calibration
Since Google Trends indices are relative, I implemented a Calibration Ratio for overlapping periods to ensure a continuous and comparable time series across multiple weeks:

Calibration Ratio = Mean(Week 1 Overlap) / Mean(Week 2 Overlap)

Week 2 Data = Calibration Ratio * Raw Week 2 Data

Normalized Index = [All Data/max(All Data)] * 100

### Time-Series Analysis
- **Pearson Correlation**: Calculate the linear relationship between physical sensing data and social sensing data.
- **Cross-Correlation (Lag Detection)**: Calculate optimal lag time between physical and social data responses to identify anticipatory, reactive, or synchronous  behaviour.
- **FFT (Frequency-Domain)**: Fast Fourier Transform to detect dominant periodic patterns (e.g., 12-hour cycle, daily cycle) in time-series data.
- **STL Decomposition**: Separate time-series into Trend, Seasonal, and Residual components for targeted anomaly analysis.


### Anomaly Detection
* **Social Anomalies:** Residuals exceeding 3 standard deviations (3-sigma).
* **Physical Anomalies:** Values exceeding mean + 1 standard deviation.
* **Association:** Events are linked if social anomalies occur within a +/- 2-hour window of a physical trigger.

## Technology
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scipy, Statsmodels, Pytrends, Plotly
* **API Integration:** Open-Meteo API (Physical Meteorological API), Google Trends API (Social Trend API)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Guoxuan Li - Imperial College London ELEC70126 - Internet of Things and Applications 2025-2026 Project
