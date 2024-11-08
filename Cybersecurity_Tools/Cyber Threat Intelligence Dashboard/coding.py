import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set the title of the dashboard
st.title("Cyber Threat Intelligence Dashboard")

# Generate mock threat data
def generate_mock_threat_data(num_entries=100):
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start="2024-01-01", periods=num_entries, freq='D')
    descriptions = [f"Threat {i}: Description of threat." for i in range(1, num_entries + 1)]
    severities = np.random.choice(['Low', 'Medium', 'High', 'Critical'], num_entries)
    latitudes = np.random.uniform(low=-90.0, high=90.0, size=num_entries)
    longitudes = np.random.uniform(low=-180.0, high=180.0, size=num_entries)
    types = np.random.choice(['Malware', 'Phishing', 'Ransomware', 'DDoS'], num_entries)

    return pd.DataFrame({
        'publishedDate': dates,
        'description': descriptions,
        'severity': severities,
        'latitude': latitudes,
        'longitude': longitudes,
        'type': types
    })

# Create mock data
df = generate_mock_threat_data()

# Display the data
st.subheader("Recent Threats")
st.dataframe(df)

# Visualization: Plotting number of threats over time
if not df.empty:
    df['date'] = pd.to_datetime(df['publishedDate'])
    threats_over_time = df.groupby(df['date'].dt.to_period('M')).size().reset_index(name='count')
    
    # Convert the Period to a string for JSON serialization
    threats_over_time['date'] = threats_over_time['date'].dt.strftime('%Y-%m')  # Format as YYYY-MM

    fig = px.line(threats_over_time, x='date', y='count', title='Threats Over Time')
    st.plotly_chart(fig)

# Search functionality
search_term = st.text_input("Search for a specific threat:")
if search_term:
    filtered_data = df[df['description'].str.contains(search_term, case=False, na=False)]
    st.dataframe(filtered_data)

# Geolocation Mapping
if 'latitude' in df.columns and 'longitude' in df.columns:
    st.subheader("Threats by Location")
    
    # Create a scatter map
    map_fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        text='description',  # Display description on hover
        title='Threats by Geolocation',
        hover_name='description',
        color='severity',  # Color by severity
        size_max=15
    )
    st.plotly_chart(map_fig)
else:
    st.warning("Geolocation data is not available.")

# Alerts Section (mock data)
def generate_mock_alerts(num_alerts=5):
    alerts = [
        {"date": f"2024-11-0{i+1}", "description": f"Critical vulnerability alert for Software {i+1}"}
        for i in range(num_alerts)
    ]
    return pd.DataFrame(alerts)

alerts_df = generate_mock_alerts()
if not alerts_df.empty:
    st.subheader("Recent Alerts")
    st.dataframe(alerts_df)

# Threat Classification
if 'severity' in df.columns:
    severity_counts = df['severity'].value_counts()
    st.subheader("Threat Classification")
    st.bar_chart(severity_counts)  # Visualize severity counts with a bar chart
else:
    st.warning("Severity data is not available.")

# User Input Filters
threat_types = df['type'].unique().tolist() if 'type' in df.columns else []
selected_type = st.selectbox("Select Threat Type", options=['All'] + threat_types)

if selected_type != 'All':
    filtered_df = df[df['type'] == selected_type]
else:
    filtered_df = df

# Display filtered data
st.dataframe(filtered_df)

# Export Data as CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_df)
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name='threat_data.csv',
    mime='text/csv',
)
