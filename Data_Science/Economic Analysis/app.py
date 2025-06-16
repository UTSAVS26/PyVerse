import streamlit as st
import pandas as pd
import plotly.express as px

# Load cleaned data
df = pd.read_csv("brics_cleaned.csv")

st.title("BRICS Economic Dashboard")
st.markdown("""
This dashboard allows you to explore key economic indicators for BRICS countries (Brazil, Russia, India, China, South Africa).  
Select a country from the sidebar to view trends, insights, and period-wise summaries.
""")

#Sidebar Country Selector
countries = df['Country'].unique()
country = st.sidebar.selectbox("Select Country", sorted(countries))

# Filter data for selected country
country_df = df[df['Country'] == country].sort_values('date')

# most recent stats
latest = country_df.sort_values('date').iloc[-1]
st.header(f"Latest Data for {country} ({latest['date'][:4]})")
st.write({
    "GDP": latest["GDP"],
    "Unemployment Rate": latest["Unemployment Rate"],
    "Inflation Rate": latest["Inflation Rate"],
    "Current Account Balance": latest["Current Account Balance"]
})

#key indicators
indicators = ['GDP', 'Unemployment Rate', 'Inflation Rate', 'Current Account Balance']
for col in indicators:
    st.subheader(f"{col} Over Time")
    fig = px.line(country_df, x='date', y=col, markers=True, title=f"{col} Over Time ({country})")
    st.plotly_chart(fig, use_container_width=True)
    # Annotate insights for specific years
    if col == 'GDP':
        max_year = country_df.loc[country_df['GDP'].idxmax()]['date'][:4]
        min_year = country_df.loc[country_df['GDP'].idxmin()]['date'][:4]
        st.info(f"Highest GDP in {max_year}. Lowest in {min_year}.")

# Year-on-Year change plots, if present
if 'GDP_YoY_Change' in country_df.columns:
    st.subheader("Year-on-Year GDP Change (%)")
    fig = px.line(country_df, x='date', y='GDP_YoY_Change', markers=True, title=f"YoY GDP Change ({country})")
    st.plotly_chart(fig, use_container_width=True)

# Period-wise summary
if "Period" in country_df.columns:
    st.subheader("Period-wise Summary")
    period_summary = country_df.groupby("Period")[indicators].mean().reset_index()
    st.dataframe(period_summary)

# Custom Insights
st.header("Insights & Recommendations")
if country == "Brazil":
    st.write("- Inflation spikes in 2016 and 2021. Policy focus: inflation control.")
    st.write("- Largest GDP fall in 2015 (-27%).")
    st.write("- Recommendation: Stabilize inflation and reduce current account deficits.")
elif country == "India":
    st.write("- GDP growth strong, but current account swings sharply (notably -1418% in 2005).")
    st.write("- Monitor inflation and trade deficits.")
elif country == "South Africa":
    st.write("- Unemployment persistently high (especially 2012–2023).")
    st.write("- Recommendation: Prioritize labor market reforms.")
elif country == "Russia":
    st.write("- GDP and inflation trends affected by external shocks.")
    st.write("- Recommendation: Diversify economy and build resilience.")
elif country == "China":
    st.write("- Consistent GDP growth. Monitor inflation in recent years.")
    st.write("- Recommendation: Continue structural reforms.")

st.caption("Data source: brics_cleaned.csv. Created with Streamlit.")
