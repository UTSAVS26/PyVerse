"""
Visualization tools for habit tracking data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class HabitVisualizer:
    """Generates visualizations for habit tracking data."""
    
    def __init__(self, tracker):
        """Initialize visualizer with habit tracker data."""
        self.tracker = tracker
        self.df = tracker.to_dataframe()
    
    def create_dashboard(self, save_path: Optional[str] = None) -> go.Figure:
        """Create a comprehensive dashboard with multiple charts."""
        if self.df.empty:
            return self._create_empty_dashboard()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Daily Habits Over Time', 'Correlation Heatmap',
                'Sleep vs Mood/Productivity', 'Exercise Impact',
                'Weekly Patterns', 'Productivity Trends'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Add traces
        self._add_daily_habits_trace(fig, row=1, col=1)
        self._add_correlation_heatmap(fig, row=1, col=2)
        self._add_sleep_analysis(fig, row=2, col=1)
        self._add_exercise_impact(fig, row=2, col=2)
        self._add_weekly_patterns(fig, row=3, col=1)
        self._add_productivity_trends(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1200,
            title_text="AI Habit Tracker Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _create_empty_dashboard(self) -> go.Figure:
        """Create an empty dashboard when no data is available."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Start logging your habits to see insights!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title_text="AI Habit Tracker Dashboard",
            template="plotly_white"
        )
        return fig
    
    def _add_daily_habits_trace(self, fig: go.Figure, row: int, col: int) -> None:
        """Add daily habits over time trace."""
        if self.df.empty:
            return
        
        # Normalize data for better visualization
        df_normalized = self.df.copy()
        df_normalized['sleep_normalized'] = df_normalized['sleep_hours'] / 10
        df_normalized['exercise_normalized'] = df_normalized['exercise_minutes'] / 120
        df_normalized['screen_normalized'] = df_normalized['screen_time_hours'] / 10
        df_normalized['mood_normalized'] = df_normalized['mood_rating'] / 5
        df_normalized['productivity_normalized'] = df_normalized['productivity_rating'] / 5
        
        fig.add_trace(
            go.Scatter(
                x=df_normalized['date'],
                y=df_normalized['sleep_normalized'],
                mode='lines+markers',
                name='Sleep (hrs/10)',
                line=dict(color='blue')
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_normalized['date'],
                y=df_normalized['exercise_normalized'],
                mode='lines+markers',
                name='Exercise (min/120)',
                line=dict(color='green')
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_normalized['date'],
                y=df_normalized['mood_normalized'],
                mode='lines+markers',
                name='Mood (rating/5)',
                line=dict(color='orange')
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_normalized['date'],
                y=df_normalized['productivity_normalized'],
                mode='lines+markers',
                name='Productivity (rating/5)',
                line=dict(color='red')
            ),
            row=row, col=col
        )
    
    def _add_correlation_heatmap(self, fig: go.Figure, row: int, col: int) -> None:
        """Add correlation heatmap trace."""
        if self.df.empty:
            return
        
        numeric_cols = ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                       'water_glasses', 'work_hours', 'mood_rating', 'productivity_rating']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation")
            ),
            row=row, col=col
        )
    
    def _add_sleep_analysis(self, fig: go.Figure, row: int, col: int) -> None:
        """Add sleep analysis trace."""
        if self.df.empty:
            return
        
        # Group sleep hours into bins
        sleep_bins = pd.cut(self.df['sleep_hours'], bins=[0, 6, 7, 8, 9, 24])
        sleep_analysis = self.df.groupby(sleep_bins).agg({
            'mood_rating': 'mean',
            'productivity_rating': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=[str(x) for x in sleep_analysis['sleep_hours']],
                y=sleep_analysis['mood_rating'],
                name='Avg Mood',
                marker_color='lightblue'
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Bar(
                x=[str(x) for x in sleep_analysis['sleep_hours']],
                y=sleep_analysis['productivity_rating'],
                name='Avg Productivity',
                marker_color='lightgreen'
            ),
            row=row, col=col
        )
    
    def _add_exercise_impact(self, fig: go.Figure, row: int, col: int) -> None:
        """Add exercise impact analysis trace."""
        if self.df.empty:
            return
        
        # Exercise vs no exercise
        exercise_days = self.df[self.df['exercise_minutes'] > 0]
        no_exercise_days = self.df[self.df['exercise_minutes'] == 0]
        
        categories = ['Exercise Days', 'No Exercise Days']
        mood_values = [
            exercise_days['mood_rating'].mean() if len(exercise_days) > 0 else 0,
            no_exercise_days['mood_rating'].mean() if len(no_exercise_days) > 0 else 0
        ]
        productivity_values = [
            exercise_days['productivity_rating'].mean() if len(exercise_days) > 0 else 0,
            no_exercise_days['productivity_rating'].mean() if len(no_exercise_days) > 0 else 0
        ]
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=mood_values,
                name='Avg Mood',
                marker_color='orange'
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=productivity_values,
                name='Avg Productivity',
                marker_color='purple'
            ),
            row=row, col=col
        )
    
    def _add_weekly_patterns(self, fig: go.Figure, row: int, col: int) -> None:
        """Add weekly patterns trace."""
        if self.df.empty:
            return
        
        # Add day of week
        df_with_dow = self.df.copy()
        df_with_dow['day_of_week'] = df_with_dow['date'].dt.day_name()
        
        weekly_patterns = df_with_dow.groupby('day_of_week').agg({
            'mood_rating': 'mean',
            'productivity_rating': 'mean'
        }).reset_index()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_patterns['day_of_week'] = pd.Categorical(weekly_patterns['day_of_week'], categories=day_order, ordered=True)
        weekly_patterns = weekly_patterns.sort_values('day_of_week')
        
        fig.add_trace(
            go.Scatter(
                x=weekly_patterns['day_of_week'],
                y=weekly_patterns['mood_rating'],
                mode='lines+markers',
                name='Avg Mood',
                line=dict(color='blue')
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=weekly_patterns['day_of_week'],
                y=weekly_patterns['productivity_rating'],
                mode='lines+markers',
                name='Avg Productivity',
                line=dict(color='red')
            ),
            row=row, col=col
        )
    
    def _add_productivity_trends(self, fig: go.Figure, row: int, col: int) -> None:
        """Add productivity trends trace."""
        if self.df.empty:
            return
        
        # Calculate rolling average
        df_with_rolling = self.df.copy()
        df_with_rolling['productivity_rolling_avg'] = df_with_rolling['productivity_rating'].rolling(window=3, min_periods=1).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df_with_rolling['date'],
                y=df_with_rolling['productivity_rating'],
                mode='markers',
                name='Daily Productivity',
                marker=dict(color='lightblue', size=8)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_with_rolling['date'],
                y=df_with_rolling['productivity_rolling_avg'],
                mode='lines',
                name='3-Day Rolling Avg',
                line=dict(color='red', width=3)
            ),
            row=row, col=col
        )
    
    def create_correlation_heatmap(self, save_path: Optional[str] = None) -> go.Figure:
        """Create a correlation heatmap."""
        if self.df.empty:
            return self._create_empty_plot("No data available for correlation analysis")
        
        numeric_cols = ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                       'water_glasses', 'work_hours', 'mood_rating', 'productivity_rating']
        
        correlation_matrix = self.df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation Coefficient")
        ))
        
        fig.update_layout(
            title="Habit Correlations",
            width=800,
            height=600,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_trend_analysis(self, save_path: Optional[str] = None) -> go.Figure:
        """Create trend analysis charts."""
        if self.df.empty:
            return self._create_empty_plot("No data available for trend analysis")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sleep Trends', 'Exercise Trends', 'Mood Trends', 'Productivity Trends'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sleep trends
        fig.add_trace(
            go.Scatter(x=self.df['date'], y=self.df['sleep_hours'], 
                      mode='lines+markers', name='Sleep Hours'),
            row=1, col=1
        )
        
        # Exercise trends
        fig.add_trace(
            go.Scatter(x=self.df['date'], y=self.df['exercise_minutes'], 
                      mode='lines+markers', name='Exercise Minutes'),
            row=1, col=2
        )
        
        # Mood trends
        fig.add_trace(
            go.Scatter(x=self.df['date'], y=self.df['mood_rating'], 
                      mode='lines+markers', name='Mood Rating'),
            row=2, col=1
        )
        
        # Productivity trends
        fig.add_trace(
            go.Scatter(x=self.df['date'], y=self.df['productivity_rating'], 
                      mode='lines+markers', name='Productivity Rating'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, width=1000, title_text="Habit Trends Over Time")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_weekly_summary(self, save_path: Optional[str] = None) -> go.Figure:
        """Create weekly summary charts."""
        if self.df.empty:
            return self._create_empty_plot("No data available for weekly analysis")
        
        # Add day of week
        df_with_dow = self.df.copy()
        df_with_dow['day_of_week'] = df_with_dow['date'].dt.day_name()
        
        weekly_summary = df_with_dow.groupby('day_of_week').agg({
            'sleep_hours': 'mean',
            'exercise_minutes': 'mean',
            'screen_time_hours': 'mean',
            'mood_rating': 'mean',
            'productivity_rating': 'mean'
        }).round(2)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_summary = weekly_summary.reindex(day_order)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Weekly Sleep Patterns', 'Weekly Exercise Patterns', 
                          'Weekly Mood Patterns', 'Weekly Productivity Patterns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sleep patterns
        fig.add_trace(
            go.Bar(x=weekly_summary.index, y=weekly_summary['sleep_hours'], 
                  name='Avg Sleep Hours'),
            row=1, col=1
        )
        
        # Exercise patterns
        fig.add_trace(
            go.Bar(x=weekly_summary.index, y=weekly_summary['exercise_minutes'], 
                  name='Avg Exercise Minutes'),
            row=1, col=2
        )
        
        # Mood patterns
        fig.add_trace(
            go.Bar(x=weekly_summary.index, y=weekly_summary['mood_rating'], 
                  name='Avg Mood Rating'),
            row=2, col=1
        )
        
        # Productivity patterns
        fig.add_trace(
            go.Bar(x=weekly_summary.index, y=weekly_summary['productivity_rating'], 
                  name='Avg Productivity Rating'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, width=1000, title_text="Weekly Habit Patterns")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    def save_all_charts(self, output_dir: str = "charts") -> Dict[str, str]:
        """Save all charts to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Dashboard
        dashboard = self.create_dashboard()
        dashboard_path = os.path.join(output_dir, "dashboard.html")
        dashboard.write_html(dashboard_path)
        saved_files['dashboard'] = dashboard_path
        
        # Correlation heatmap
        corr_heatmap = self.create_correlation_heatmap()
        corr_path = os.path.join(output_dir, "correlation_heatmap.html")
        corr_heatmap.write_html(corr_path)
        saved_files['correlation_heatmap'] = corr_path
        
        # Trend analysis
        trends = self.create_trend_analysis()
        trends_path = os.path.join(output_dir, "trend_analysis.html")
        trends.write_html(trends_path)
        saved_files['trend_analysis'] = trends_path
        
        # Weekly summary
        weekly = self.create_weekly_summary()
        weekly_path = os.path.join(output_dir, "weekly_summary.html")
        weekly.write_html(weekly_path)
        saved_files['weekly_summary'] = weekly_path
        
        return saved_files
