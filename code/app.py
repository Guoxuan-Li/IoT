# %% app.py
# streamlit run C:\Users\MIA\Desktop\app.py  change to your path to run the code
# Including time domain, frequency domain, STL decomposition and click interaction

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, fftfreq

# configuration
github_data = {
    'Traffic': 'https://raw.githubusercontent.com/Guoxuan-Li/IoT/refs/heads/main/data/processed/Traffic_final.csv',
    'Weather': 'https://raw.githubusercontent.com/Guoxuan-Li/IoT/refs/heads/main/data/processed/Weather_final.csv'
}

st.set_page_config(page_title="London CPSS Dashboard", layout="wide", page_icon="🌦️")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "System Overview"

# Top UI (title)
st.title("Cyber-Physical-Social System based on London")
st.markdown("**An automated system that correlates specific search keywords with local weather data to analyze social responses to environmental changes.**")
st.divider()

# Back-end engine
@st.cache_resource
def get_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502])
    s.mount('https://', HTTPAdapter(max_retries=retries))
    return s

# Data standardization and Cleaning
@st.cache_data(ttl=3600)
def load_social_data(url):
    try:
        df = pd.read_csv(url)
        if 'timestamp' not in df.columns: 
            df.rename(columns={df.columns[0]: 'timestamp', df.columns[1]: 'social_index'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None) 
        df.set_index('timestamp', inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def weather_data(start_t, end_t):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 51.5074, "longitude": -0.1278,
        "start_date": start_t.strftime('%Y-%m-%d'),
        "end_date": end_t.strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "precipitation", "rain", "wind_speed_10m"],
        "timezone": "Europe/London"
    }
    try:
        resp = get_session().get(url, params=params)
        w_df = pd.DataFrame(resp.json()['hourly'])
        w_df['timestamp'] = pd.to_datetime(w_df['time'])
        if w_df['timestamp'].dt.tz is not None:
             w_df['timestamp'] = w_df['timestamp'].dt.tz_localize(None)
        w_df.set_index('timestamp', inplace=True)
        return w_df.loc[start_t:end_t], None
    except Exception as e:
        return None, str(e)

# Side control panel
with st.sidebar:
    st.header("System Controls")
    soc_metric = st.selectbox("Social API (Google Trends):", list(github_data.keys()))
    phys_metric = st.selectbox("Physical API (Open-Meteo):", ["precipitation", "rain", "temperature_2m", "wind_speed_10m"])
    window = st.slider("Low-Pass Filter Window (Hrs):", 1, 12, 3, help="Applies a moving average to smooth high-frequency noise.")
    
    unit = ""
    if "temp" in phys_metric: unit = " °C"
    elif "speed" in phys_metric: unit = " km/h"
    elif "precip" in phys_metric or "rain" in phys_metric: unit = " mm"

# Main logic
with st.spinner("Initiating Data Fusion Pipeline & DSP Engine..."):
    full_social_df, s_err = load_social_data(github_data[soc_metric])

if full_social_df is not None:
    # time control
    with st.sidebar:
        st.divider()
        st.subheader("Analysis Window")
        
        abs_start = full_social_df.index.min().date()
        abs_end = full_social_df.index.max().date()
        
        # select time
        date_range = st.date_input(
            "Select Range:",
            value=(abs_start, abs_end),
            min_value=abs_start,
            max_value=abs_end
        )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        start_t = pd.to_datetime(start_date)
        end_t = pd.to_datetime(end_date) + pd.Timedelta(hours=23, minutes=59)
        
        social_df = full_social_df.loc[start_t:end_t]
        
        if len(social_df) < 48:
            st.warning("Data too short for DSP analysis. Select at least 48h.")
            st.stop()
            
        with st.sidebar:
            st.success(f"Filtering: {len(social_df)} data points.")
    else:
        st.info("Please select a start and end date on the calendar.")
        st.stop()
    weather_df, w_err = weather_data(start_t, end_t)
    
    if weather_df is not None:
        # Align the two datasets forcibly according to the timestamps
        df = pd.merge(weather_df, social_df, left_index=True, right_index=True, how='inner')
        # DSP
        df['phys_smooth'] = df[phys_metric].rolling(window, center=True).mean()
        df['social_smooth'] = df['social_index'].rolling(window, center=True).mean()
        # Remove invalid data
        df.dropna(inplace=True)
        
        # Time-domain analysis (Lag & Correlation)
        # Pearson correlation coefficient
        corr = df['phys_smooth'].corr(df['social_smooth'])
        # Define lag range
        lags = np.arange(-24, 25)
        # Find the lag
        corrs = [df['social_smooth'].corr(df['phys_smooth'].shift(lag)) for lag in lags]
        best_lag = lags[np.argmax(np.abs(corrs))]
        
        # FFT calculation
        N = len(df)
        T = 1
        # Subtract the mean value and force the analysis to focus on the frequency of changes
        soc_fft = fft(df['social_smooth'].values - df['social_smooth'].mean())
        phys_fft = fft(df['phys_smooth'].values - df['phys_smooth'].mean())
        xf = fftfreq(N, T)[:N//2]
        periods = 1.0 / xf[xf > 0]
        soc_pwr = np.abs(soc_fft[0:N//2][xf > 0 ])/N
        phys_pwr = np.abs(phys_fft[0:N//2][xf > 0])/N
        dom_soc_period = periods[np.argmax(soc_pwr)]
        dom_phys_period = periods[np.argmax(phys_pwr)]

        # STL calculation
        try:
            period_val = int(round(dom_soc_period)) if dom_soc_period > 0 else 24
            res = STL(df['social_smooth'], period=period_val, robust=True).fit()
        except:
            period_val = 24
            res = STL(df['social_smooth'], period=24, robust=True).fit()
        resid_std = res.resid.std()
        # higher than 3std
        anomalies = res.resid[res.resid.abs() > 3 * resid_std]

        # mean + std
        phys_threshold = df['phys_smooth'].mean() + df['phys_smooth'].std()
        # physical anomalies
        phys_active_times = df[df['phys_smooth'] > phys_threshold].index

        phys_impact_window = pd.DatetimeIndex([]) 
        for t in phys_active_times:
            w = pd.date_range(start=t - pd.Timedelta(hours=1), end=t + pd.Timedelta(hours=1), freq='h')
            phys_impact_window = phys_impact_window.union(w)

        # the intersection of physical anomalies and social anomalies
        verified_a = anomalies.index.intersection(phys_impact_window)


        # 5 Pages
        tab_titles= [
            "1. System Overview",
            "2. Temporal Dynamics", 
            "3. Cross-Correlation & Lags", 
            "4. Periodicity and Anomaly Detection", 
            "5. Processed Data"
        ]
        active_tab = st.radio("Navigation", tab_titles, horizontal=True, label_visibility="collapsed", key="active_tab_state")
        st.divider()

        # TAB 1: overview
        if active_tab == "1. System Overview":
            st.markdown("### System Overview")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Pearson Correlation (r)", f"{corr:.3f}")
            k2.metric("Optimal Response Lag", f"{best_lag}h")
            k3.metric("Social Behavior Cycle", f"{dom_soc_period:.1f}h")
            k4.metric("Data Size", f"{len(df)}")

            st.divider()
            if best_lag > 0:
                mode, article, status = "Reactive", "a", "warning" 
            elif best_lag < 0:
                mode, article, status = "Anticipatory", "an", "success" 
            else:
                mode, article, status = "Synchronous", "a", "info"  

            insight_msg = f"The cross-correlation peaks at **{best_lag} hours**, indicating {article} **{mode}** modeling framework."

            if status == "warning":
                st.warning(insight_msg)
            elif status == "success":
                st.success(insight_msg)
            else:
                st.info(insight_msg)


        
            c_left, c_right = st.columns([1, 2])
            with c_left:
                with st.container(border=True):
                    st.markdown("**Event Localization Map**")
                    # London Location
                    map_df = pd.DataFrame({'lat': [51.5074], 'lon': [-0.1278]})
                    st.map(map_df, zoom=10, height=220)

                with st.container(border=True):
                    st.markdown("**Anomalies**")

                    if not verified_a.empty:
                        dt_index = pd.to_datetime(verified_a).unique().sort_values()
                        formatted_times = dt_index.strftime('%m-%d %H:%M')
                        # Physical abnormalities have led to the emergence of social abnormalities.
                        st.error(f" Detected {len(verified_a)} physical-social abnormal correlation events.")
                        # happend time
                        st.caption(f"Impact Times: {', '.join(formatted_times)}")
                    elif not anomalies.empty:
                        # abnormality in social behavior, but it is not caused by the physical weather
                        st.warning(f"{len(anomalies)} anomalies detected, but likely unrelated to {phys_metric}.")
                    else:
                        st.success("System operating within normal 3σ limits.")
            
            with c_right:
                with st.container(border=True):
                    st.markdown("**Global Trend Preview**")
                    
                    fig_mini = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # physical index
                    fig_mini.add_trace(go.Scatter(
                        x=df.index, y=df['phys_smooth'], 
                        name=f"Physical ({phys_metric})", 
                        line=dict(color='blue', width=2)
                    ), secondary_y=False)
                    
                    # social index
                    fig_mini.add_trace(go.Scatter(
                        x=df.index, y=df['social_smooth'], 
                        name=f"Social ({soc_metric})", 
                        line=dict(color='red', width=2, dash='dot')
                    ), secondary_y=True)

                    fig_mini.update_layout(
                        height=350, 
                        margin=dict(l=0, r=0, t=10, b=0), 
                        hovermode="x unified"
                    )
                    fig_mini.update_yaxes(title_text=f"Physical Data ({unit})", secondary_y=False)
                    fig_mini.update_yaxes(title_text="Relative Interest (0-100)", secondary_y=True)
                    
                    st.plotly_chart(fig_mini, use_container_width=True, key="mini_map_overview")


        # TAB 2: Core Time Series Interaction Diagram
        elif active_tab =="2. Temporal Dynamics":
            st.markdown("#### Temporal Dynamics")
            mode_col, reset_col = st.columns([4, 1])
            with mode_col:
                interaction_mode = st.radio(
                    "Interaction Mode:", 
                    ["Inspect Point (Click)", "Analyze Region (Box)"], 
                    horizontal=True, label_visibility="collapsed"
                )
            with reset_col:
                if st.button("Reset View", use_container_width=True):
                    st.rerun()

        
            # Plot the main graph
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            # Physics data
            fig.add_trace(go.Scatter(
                x=df.index, y=df['phys_smooth'], name=f"Physical ({phys_metric})",
                mode='lines+markers', marker=dict(size=4, color='blue'), line=dict(width=2),
                hovertemplate=f"%{{y:.2f}}{unit}<extra></extra>"
            ), secondary_y=False)

            # Social data 
            fig.add_trace(go.Scatter(
                x=df.index, y=df['social_smooth'], name=f"Social ({soc_metric})",
                mode='lines+markers', line=dict(width=2, dash='dot', color='red'),
                hovertemplate="%{y:.2f}<extra></extra>"
            ), secondary_y=True)

            d_mode = "select" if "Region" in interaction_mode else "pan"
            fig.update_layout(
                height=450, 
                hovermode="x unified", 
                dragmode=d_mode, 
                clickmode='event+select',
                margin=dict(l=0, r=0, t=30, b=0))
            fig.update_yaxes(title_text=f"Physical Data ({unit})", secondary_y=False)
            fig.update_yaxes(title_text="Relative Interest (0-100)", secondary_y=True)
            
            chart_key = f"chart_{soc_metric}_{phys_metric}_{interaction_mode}"
            event_t1 = st.plotly_chart(
                fig, 
                use_container_width=True, 
                on_select="rerun", 
                selection_mode=["points", "box"], 
                key=chart_key
            )
            
            if event_t1 and "selection" in event_t1 and event_t1["selection"]["points"]:
                points = event_t1["selection"]["points"]
                
                try:
                    selected_times = pd.to_datetime([p['x'] for p in points], format='mixed')
                except:
                    selected_times = pd.to_datetime([p['x'] for p in points], errors='coerce')
                
                selected_times = selected_times[selected_times.notnull()]
                
                if len(selected_times) > 0:
                    if len(selected_times) == 1:
                        # point
                        target_time = selected_times[0]
                        idx = df.index.get_indexer([target_time], method='nearest')[0]
                        row = df.iloc[idx]
                        actual_time = row.name
                        
                        st.markdown(f"### Detailed Information: {actual_time.strftime('%Y-%m-%d %H:%M')}")

                        c1, c2, c3 = st.columns(3)
    
                        with c1:
                            st.info(f"**Time** \n ### {actual_time.strftime('%H:%M')}")
    
                        with c2:
                            st.success(f"**{phys_metric}** \n ### {row['phys_smooth']:.2f}{unit}")

                        with c3:
                            st.warning(f"**{soc_metric}** \n ### {row['social_smooth']:.2f}")

                    elif len(selected_times) > 1:
                        # Box
                        t_min, t_max = selected_times.min(), selected_times.max()
                        mask_df = df.loc[t_min:t_max]
                        
                        if not mask_df.empty:
                            st.markdown(f"### Region Stats ({len(mask_df)}h window)")
                            st.caption(f"**From:** {t_min.strftime('%Y-%m-%d %H:%M')} &nbsp;&nbsp;|&nbsp;&nbsp; **To:** {t_max.strftime('%Y-%m-%d %H:%M')}")
                            c1, c2, c3 = st.columns(3)

                            with c1: 
                                st.info(f"**Avg {phys_metric}** \n ### {mask_df['phys_smooth'].mean():.2f}{unit}")

                            with c2:
                                st.success(f"**Avg {soc_metric}** \n ### {mask_df['social_smooth'].mean():.2f}")

                            with c3:
                                std_phys = mask_df['phys_smooth'].std()
                                std_soc = mask_df['social_smooth'].std()
                                    
                                if std_phys == 0:
                                    st.error(f"**Correlation** \n\n NaN: Constant {phys_metric}")
                                elif std_soc == 0:
                                    st.error(f"**Correlation** \n\n NaN: Constant {soc_metric}")
                                else:
                                    current_corr = mask_df['phys_smooth'].corr(mask_df['social_smooth'])
                                    st.warning(f"**Correlation** \n ### {current_corr:.2f}")
            else:
                st.caption("Tips: Switch to 'Analyze Region' above to drag a box on the chart.")


        # TAB 3: Cross-Correlation & Lags
        elif active_tab =="3. Cross-Correlation & Lags":
            st.markdown("#### Cross-Correlation & Lags")
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Pearson Correlation (r)", f"{corr:.3f}")
            k2.metric("Optimal Response Lag", f"{best_lag}h")
            k3.metric("Data Size", f"{len(df)}")

            max_corr = max(corrs)
            if best_lag > 0:
                interpretation = f"The social response ({soc_metric}) typically **lags behind** physical events ({phys_metric}) by {best_lag} hours."
            elif best_lag < 0:
                interpretation = f"The social sensing ({soc_metric}) **anticipates** physical events ({phys_metric}) by {abs(best_lag)} hours."
            else:
                interpretation = f"The physical ({phys_metric}) and social ({soc_metric}) systems are **synchronized**."

            st.success(f"{interpretation} (Peak Correlation r = {max_corr:.3f})")


            col1, col2 = st.columns(2)
            margin = dict(l=20, r=100, t=60, b=40)
            
            with col1: 
                # scatter plot
                fig_scatter = go.Figure()
                
                # data
                fig_scatter.add_trace(go.Scatter(
                    x=df['phys_smooth'], y=df['social_smooth'], 
                    mode='markers', name="Data Samples",
                    marker=dict(color='purple', opacity=0.4, size=6)
                ))
                
                # best fit line
                z = np.polyfit(df['phys_smooth'], df['social_smooth'], 1)
                p = np.poly1d(z)
                fig_scatter.add_trace(go.Scatter(
                    x=df['phys_smooth'], y=p(df['phys_smooth']),
                    mode='lines', name="Trendline",
                    line=dict(color='blue', width=2, dash='dash')
                ))

                fig_scatter.update_layout(
                    title="Correlation Distribution", 
                    height=400, 
                    margin=margin,
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=0.98)
                )
                fig_scatter.update_xaxes(title=f"Physical ({phys_metric})")
                fig_scatter.update_yaxes(title="Social Index")
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption("The slope indicates the **Responsiveness** of social sensing to physical environment.")

            with col2: 
                # lag
                colors = ['red' if lag == best_lag else 'lightgray' for lag in lags]
                fig_lag = go.Figure(data=go.Bar(
                    x=lags, y=corrs, 
                    marker_color=colors,
                    name="Correlation"
                ))
                

                fig_lag.update_layout(
                    title="Cross-Correlation Function (CCF)", 
                    height=400, 
                    margin=margin
                )
                fig_lag.update_xaxes(title="Lag Time (Hours)")
                fig_lag.update_yaxes(title="Correlation Coefficient (r)")
                st.plotly_chart(fig_lag, use_container_width=True)
                st.caption("The red bar identifies the **Reaction Time** of the CPSS system.")

        # TAB 4: FFT & STL
        elif active_tab =="4. Periodicity and Anomaly Detection":
        
            st.markdown("#### Fourier Transform Analysis (FFT)")
            m1, m2 = st.columns(2)
            m1.metric(f"Social Cycle ({soc_metric})", f"{dom_soc_period:.1f}h")
            m2.metric(f"Physical Cycle ({phys_metric})", f"{dom_phys_period:.1f}h")

            # FFT 
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(x=periods, y=soc_pwr, mode='lines', name='Social Spectrum', line=dict(color='red')))
            fig_fft.add_trace(go.Scatter(x=periods, y=phys_pwr, mode='lines', name='Physical Spectrum', line=dict(color='blue')))
            
            fig_fft.add_vline(x=12, line_dash="dot", line_color="rgba(0,0,0,0.2)", annotation_text="Half-Day")
            fig_fft.add_vline(x=24, line_dash="dash", line_color="rgba(0,0,0,0.3)", annotation_text="Daily")
            fig_fft.update_layout(
                title="Power Spectral Density (Period)", height=350, 
                xaxis_title="Period (Hours / Cycle)", yaxis_title="Amplitude",
                xaxis=dict(range=[0, 72])
            )

            fft_key = f"fft_plot_{soc_metric}_{phys_metric}"
            st.plotly_chart(fig_fft, use_container_width=True, key=fft_key)
            st.divider()

            st.markdown("#### Time-Series Decomposition (STL)")         
            if not anomalies.empty:
                st.error(f"{len(anomalies)} anomalies detected in the Residual component. And {len(verified_a)} events were related to physical changes.")
            
            fig_stl = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", f"Seasonal ({period_val}h)", "Residuals"))
            fig_stl.add_trace(go.Scatter(x=df.index, y=res.trend, name="Trend"), 1, 1)
            fig_stl.add_trace(go.Scatter(x=df.index, y=res.seasonal, name="Seasonal"), 2, 1)
            fig_stl.add_trace(go.Scatter(x=df.index, y=res.resid, mode='markers', name="Residuals", marker=dict(color='lightgray', size=4)), 3, 1)
            first_vrect = True
            for t in phys_active_times:
                fig_stl.add_vrect(
                    x0=t - pd.Timedelta(hours=0.5), x1=t + pd.Timedelta(hours=0.5),
                    fillcolor="purple", opacity=0.1, layer="below", line_width=0,
                    row=3, col=1, name=f"High {phys_metric}",
                    showlegend=True if first_vrect else False
                )
                first_vrect = False

            if not anomalies.empty:
                fig_stl.add_trace(go.Scatter(
                    x=anomalies.index, y=anomalies.values,
                    mode='markers', name="Detected Shocks",
                    marker=dict(color='red', size=8, symbol='x')
                ), 3, 1)
                fig_stl.add_hline(y=3*resid_std, line_dash="dash", line_color="red", row=3, col=1, opacity=0.3)
                fig_stl.add_hline(y=-3*resid_std, line_dash="dash", line_color="red", row=3, col=1, opacity=0.3)

            fig_stl.update_layout(height=700, showlegend=True, margin=dict(t=50))
            stl_key = f"stl_plot_{soc_metric}_{phys_metric}"
            st.plotly_chart(fig_stl, use_container_width=True, key=stl_key)


        # TAB 5: Data file
        elif active_tab == "5. Processed Data":
            st.markdown("#### Processed Data")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Dataset (CSV)",
                data=csv,
                file_name=f"London_Index_{phys_metric}_{soc_metric}.csv",
                mime='text/csv',
            )

    else:
        st.error(f"Weather API Error: {w_err}")
else:
    st.error(f"GitHub Error: {s_err}")