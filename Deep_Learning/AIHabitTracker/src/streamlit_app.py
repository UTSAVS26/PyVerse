"""
Streamlit web application for AI Habit Tracker.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import numpy as np

from src.models.database import DatabaseManager
from src.models.habit_model import HabitEntry
from src.analysis.pattern_detector import PatternDetector
from src.analysis.visualizer import HabitVisualizer


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Habit Tracker",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 AI Habit Tracker with Pattern Detection")
    st.markdown("Track your daily habits and discover AI-powered insights!")
    
    # Initialize database and components
    db_manager = DatabaseManager()
    tracker = db_manager.load_tracker()
    pattern_detector = PatternDetector(tracker)
    visualizer = HabitVisualizer(tracker)
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📝 Log Habits", "📊 Dashboard", "🔍 AI Insights", "📈 Analytics", "💾 Data Management"]
    )
    
    if page == "📝 Log Habits":
        show_logging_page(db_manager, tracker)
    elif page == "📊 Dashboard":
        show_dashboard_page(tracker, visualizer)
    elif page == "🔍 AI Insights":
        show_insights_page(pattern_detector)
    elif page == "📈 Analytics":
        show_analytics_page(tracker, visualizer)
    elif page == "💾 Data Management":
        show_data_management_page(db_manager, tracker)


def show_logging_page(db_manager, tracker):
    """Show the habit logging page."""
    st.header("📝 Daily Habit Logger")
    
    # Create form
    with st.form("habit_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Date selection
            entry_date = st.date_input("Date", value=date.today())
            
            # Habit inputs
            st.subheader("Daily Habits")
            sleep_hours = st.number_input("Sleep (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            exercise_minutes = st.number_input("Exercise (minutes)", min_value=0, value=30, step=5)
            screen_time_hours = st.number_input("Screen time (hours)", min_value=0.0, value=4.0, step=0.5)
            water_glasses = st.number_input("Water (glasses)", min_value=0, value=8, step=1)
            work_hours = st.number_input("Work hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
        
        with col2:
            # Ratings
            st.subheader("Daily Ratings (1-5)")
            mood_rating = st.slider("Mood", min_value=1, max_value=5, value=3, step=1)
            productivity_rating = st.slider("Productivity", min_value=1, max_value=5, value=3, step=1)
            
            # Notes
            st.subheader("Notes (Optional)")
            notes = st.text_area("Add any notes about your day", height=100)
        
        # Submit button
        submitted = st.form_submit_button("Save Entry")
        
        if submitted:
            try:
                entry = HabitEntry(
                    date=entry_date,
                    sleep_hours=sleep_hours,
                    exercise_minutes=exercise_minutes,
                    screen_time_hours=screen_time_hours,
                    water_glasses=water_glasses,
                    work_hours=work_hours,
                    mood_rating=mood_rating,
                    productivity_rating=productivity_rating,
                    notes=notes
                )
                
                if db_manager.save_entry(entry):
                    tracker.add_entry(entry)
                    st.success("✅ Habit entry saved successfully!")
                else:
                    st.error("❌ Failed to save entry!")
                    
            except ValueError as e:
                st.error(f"❌ Validation Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Show recent entries
    st.subheader("Recent Entries")
    entries = db_manager.get_all_entries()
    if entries:
        df = pd.DataFrame([entry.to_dict() for entry in entries[-10:]])
        df['date'] = pd.to_datetime(df['date'])
        st.dataframe(df[['date', 'sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                        'water_glasses', 'work_hours', 'mood_rating', 'productivity_rating']])
    else:
        st.info("No entries yet. Start logging your habits!")


def show_dashboard_page(tracker, visualizer):
    """Show the main dashboard page."""
    st.header("📊 Habit Dashboard")
    
    # Summary statistics
    stats = tracker.get_summary_stats()
    
    if not stats:
        st.info("No data available. Start logging your habits to see insights!")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entries", stats['total_entries'])
    
    with col2:
        st.metric("Avg Sleep", f"{stats['averages']['sleep_hours']:.1f} hrs")
    
    with col3:
        st.metric("Avg Exercise", f"{stats['averages']['exercise_minutes']:.0f} min")
    
    with col4:
        st.metric("Avg Mood", f"{stats['averages']['mood_rating']:.1f}/5")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Habits Over Time")
        df = tracker.to_dataframe()
        if not df.empty:
            fig = px.line(df, x='date', y=['sleep_hours', 'exercise_minutes', 'screen_time_hours'],
                         title="Habit Trends")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Mood & Productivity")
        if not df.empty:
            fig = px.scatter(df, x='mood_rating', y='productivity_rating', 
                           title="Mood vs Productivity")
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Habit Correlations")
    if not df.empty:
        numeric_cols = ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                       'water_glasses', 'work_hours', 'mood_rating', 'productivity_rating']
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, 
                       title="Correlation Heatmap",
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)


def show_insights_page(pattern_detector):
    """Show the AI insights page."""
    st.header("🔍 AI-Powered Insights")
    
    # Get insights
    insights = pattern_detector.generate_insights()
    recommendations = pattern_detector.get_recommendations()
    patterns = pattern_detector.detect_patterns()
    predictions = pattern_detector.predict_productivity()
    
    # Key Insights
    st.subheader("💡 Key Insights")
    for insight in insights:
        st.write(f"• {insight}")
    
    # Recommendations
    st.subheader("🎯 Recommendations")
    for rec in recommendations:
        st.write(f"• {rec}")
    
    # Pattern Analysis
    st.subheader("📈 Pattern Analysis")
    if isinstance(patterns, dict) and 'message' in patterns:
        st.info(patterns['message'])
    else:
        for pattern_type, pattern_data in patterns.items():
            if pattern_data:
                st.write(f"**{pattern_type.title()} Patterns:**")
                if isinstance(pattern_data, dict):
                    for key, value in pattern_data.items():
                        st.write(f"  - {key}: {value}")
                st.write("")
    
    # Predictions
    st.subheader("🔮 Predictions")
    if 'message' in predictions:
        st.info(predictions['message'])
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Productivity", f"{predictions['predicted_productivity']}/5")
        with col2:
            st.metric("Confidence", f"{predictions['confidence']:.2f}")
        
        st.write("**Feature Importance:**")
        for feature, importance in predictions['feature_importance'].items():
            st.write(f"  - {feature}: {importance:.3f}")


def show_analytics_page(tracker, visualizer):
    """Show the analytics page."""
    st.header("📈 Advanced Analytics")
    
    df = tracker.to_dataframe()
    if df.empty:
        st.info("No data available for analytics.")
        return
    
    # Chart options
    chart_type = st.selectbox(
        "Choose a chart type",
        ["Trend Analysis", "Weekly Patterns", "Sleep Analysis", "Exercise Impact", "Custom Analysis"]
    )
    
    if chart_type == "Trend Analysis":
        st.subheader("Trend Analysis")
        fig = visualizer.create_trend_analysis()
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Weekly Patterns":
        st.subheader("Weekly Patterns")
        fig = visualizer.create_weekly_summary()
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Sleep Analysis":
        st.subheader("Sleep Analysis")
        # Create sleep analysis chart
        sleep_bins = pd.cut(df['sleep_hours'], bins=[0, 6, 7, 8, 9, 24])
        sleep_analysis = df.groupby(sleep_bins).agg({
            'mood_rating': 'mean',
            'productivity_rating': 'mean'
        }).reset_index()
        
        fig = px.bar(sleep_analysis, x='sleep_hours', y=['mood_rating', 'productivity_rating'],
                    title="Sleep Impact on Mood and Productivity",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Exercise Impact":
        st.subheader("Exercise Impact")
        # Exercise vs no exercise comparison
        exercise_days = df[df['exercise_minutes'] > 0]
        no_exercise_days = df[df['exercise_minutes'] == 0]
        
        if len(exercise_days) > 0 and len(no_exercise_days) > 0:
            comparison_data = pd.DataFrame({
                'Category': ['Exercise Days', 'No Exercise Days'] * 2,
                'Metric': ['Mood'] * 2 + ['Productivity'] * 2,
                'Value': [
                    exercise_days['mood_rating'].mean(),
                    no_exercise_days['mood_rating'].mean(),
                    exercise_days['productivity_rating'].mean(),
                    no_exercise_days['productivity_rating'].mean()
                ]
            })
            
            fig = px.bar(comparison_data, x='Category', y='Value', color='Metric',
                        title="Exercise Impact on Mood and Productivity",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need both exercise and no-exercise days for comparison.")
    
    elif chart_type == "Custom Analysis":
        st.subheader("Custom Analysis")
        
        # Custom chart options
        x_axis = st.selectbox("X-axis", ['date', 'sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                                        'water_glasses', 'work_hours', 'mood_rating', 'productivity_rating'])
        y_axis = st.selectbox("Y-axis", ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                                        'water_glasses', 'work_hours', 'mood_rating', 'productivity_rating'])
        
        chart_type_custom = st.selectbox("Chart type", ["scatter", "line", "bar"])
        
        if chart_type_custom == "scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        elif chart_type_custom == "line":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        else:
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
        
        st.plotly_chart(fig, use_container_width=True)


def show_data_management_page(db_manager, tracker):
    """Show the data management page."""
    st.header("💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Data")
        if st.button("Export to CSV"):
            df = tracker.to_dataframe()
            if not df.empty:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"habit_data_{date.today().isoformat()}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export.")
    
    with col2:
        st.subheader("Clear Data")
        if st.button("Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data permanently"):
                if db_manager.clear_all_data():
                    st.success("All data cleared successfully!")
                    st.rerun()
                else:
                    st.error("Failed to clear data.")
    
    # Data summary
    st.subheader("Data Summary")
    stats = db_manager.get_summary_stats()
    
    if stats.get('total_entries', 0) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Entries", stats['total_entries'])
            st.metric("Date Range", f"{stats['date_range']['start']} to {stats['date_range']['end']}")
        
        with col2:
            averages = stats['averages']
            st.metric("Avg Sleep", f"{averages['sleep_hours']:.1f} hrs")
            st.metric("Avg Exercise", f"{averages['exercise_minutes']:.0f} min")
            st.metric("Avg Screen Time", f"{averages['screen_time_hours']:.1f} hrs")
        
        with col3:
            st.metric("Avg Water", f"{averages['water_glasses']:.0f} glasses")
            st.metric("Avg Work", f"{averages['work_hours']:.1f} hrs")
            st.metric("Avg Mood", f"{averages['mood_rating']:.1f}/5")
            st.metric("Avg Productivity", f"{averages['productivity_rating']:.1f}/5")
    else:
        st.info("No data available.")
    
    # Recent entries table
    st.subheader("Recent Entries")
    entries = db_manager.get_all_entries()
    if entries:
        df = pd.DataFrame([entry.to_dict() for entry in entries[-20:]])
        df['date'] = pd.to_datetime(df['date'])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No entries found.")


if __name__ == "__main__":
    main()
