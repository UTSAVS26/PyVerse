"""
Mood Timeline Visualization Module for MoodMeet

Provides interactive timeline charts and sentiment trend visualizations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import streamlit as st


class MoodTimelineVisualizer:
    """Creates timeline visualizations for mood analysis."""
    
    def __init__(self, theme: str = "plotly"):
        """
        Initialize visualizer.
        
        Args:
            theme: Plot theme ('plotly', 'plotly_dark', 'plotly_white')
        """
        self.theme = theme
        self.colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#808080',   # Gray
            'background': '#FFFFFF',
            'grid': '#E5E5E5'
        }
    
    def create_sentiment_timeline(self, df: pd.DataFrame, 
                                text_column: str = 'text',
                                polarity_column: str = 'polarity',
                                speaker_column: Optional[str] = None) -> go.Figure:
        """
        Create interactive sentiment timeline chart.
        
        Args:
            df: DataFrame with sentiment data
            text_column: Column name for text
            polarity_column: Column name for polarity scores
            speaker_column: Optional column name for speaker
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return go.Figure()
        
        # Validate required columns
        missing = [c for c in [text_column, polarity_column] if c not in df.columns]
        if missing:
            return go.Figure()
        
        # Create timeline data
        df_copy = df.copy()
        df_copy['index'] = range(len(df_copy))
        # Create hover text
        hover_text = []
        for idx, row in df_copy.iterrows():
            text = row.get(text_column, '')
            text = text[:50] + "..." if len(text) > 50 else text
            speaker_info = f"<br>Speaker: {row[speaker_column]}" if speaker_column else ""
            hover_text.append(f"Text: {text}{speaker_info}<br>Sentiment: {row[polarity_column]:.3f}")
        
        # Create color mapping
        colors = []
        for polarity in df_copy[polarity_column]:
            if polarity > 0.1:
                colors.append(self.colors['positive'])
            elif polarity < -0.1:
                colors.append(self.colors['negative'])
            else:
                colors.append(self.colors['neutral'])
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_copy['index'],
            y=df_copy[polarity_column],
            mode='markers+lines',
            marker=dict(
                size=8,
                color=colors,
                line=dict(width=1, color='white')
            ),
            line=dict(color='lightgray', width=1),
            text=hover_text,
            hoverinfo='text',
            name='Sentiment Timeline'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title="Sentiment Timeline",
            xaxis_title="Message Sequence",
            yaxis_title="Sentiment Score",
            template=self.theme,
            hovermode='closest',
            showlegend=False
        )
        
        return fig
    
    def create_speaker_sentiment_chart(self, df: pd.DataFrame,
                                     speaker_column: str = 'speaker',
                                     polarity_column: str = 'polarity') -> go.Figure:
        """
        Create speaker-wise sentiment comparison chart.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Column name for speaker
            polarity_column: Column name for polarity scores
            
        Returns:
            Plotly figure object
        """
        if df.empty or speaker_column not in df.columns or polarity_column not in df.columns:
            return go.Figure()
        
        # Group by speaker
        speaker_stats = df.groupby(speaker_column)[polarity_column].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Create bar chart
        fig = go.Figure()
        
        # Color bars based on sentiment
        colors = []
        for mean_sentiment in speaker_stats['mean']:
            if mean_sentiment > 0.1:
                colors.append(self.colors['positive'])
            elif mean_sentiment < -0.1:
                colors.append(self.colors['negative'])
            else:
                colors.append(self.colors['neutral'])
        
        fig.add_trace(go.Bar(
            x=speaker_stats[speaker_column],
            y=speaker_stats['mean'],
            error_y=dict(type='data', array=speaker_stats['std']),
            marker_color=colors,
            text=speaker_stats['count'],
            textposition='auto',
            name='Average Sentiment'
        ))
        
        # Update layout
        fig.update_layout(
            title="Speaker Sentiment Comparison",
            xaxis_title="Speaker",
            yaxis_title="Average Sentiment Score",
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def create_sentiment_distribution(self, df: pd.DataFrame,
                                    polarity_column: str = 'polarity') -> go.Figure:
        """
        Create sentiment distribution histogram.
        
        Args:
            df: DataFrame with sentiment data
            polarity_column: Column name for polarity scores
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return go.Figure()
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df[polarity_column],
            nbinsx=20,
            marker_color='lightblue',
            opacity=0.7,
            name='Sentiment Distribution'
        ))
        
        # Add vertical line for mean
        mean_sentiment = df[polarity_column].mean()
        fig.add_vline(x=mean_sentiment, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_sentiment:.3f}")
        
        # Update layout
        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment Score",
            yaxis_title="Frequency",
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def create_moving_average_chart(self, df: pd.DataFrame,
                                  polarity_column: str = 'polarity',
                                  window_size: int = 5) -> go.Figure:
        """
        Create moving average sentiment trend chart.
        
        Args:
            df: DataFrame with sentiment data
            polarity_column: Column name for polarity scores
            window_size: Size of moving average window
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return go.Figure()
        
        df_copy = df.copy()
        df_copy['index'] = range(len(df_copy))
        
        # Calculate moving average
        df_copy['moving_avg'] = df_copy[polarity_column].rolling(
            window=window_size, center=True
        ).mean()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sentiment Timeline', 'Moving Average Trend'),
            vertical_spacing=0.1
        )
        
        # Original sentiment
        fig.add_trace(
            go.Scatter(
                x=df_copy['index'],
                y=df_copy[polarity_column],
                mode='markers',
                marker=dict(size=4, color='lightblue'),
                name='Individual Sentiment',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Moving average
        fig.add_trace(
            go.Scatter(
                x=df_copy['index'],
                y=df_copy['moving_avg'],
                mode='lines',
                line=dict(color='red', width=3),
                name='Moving Average',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title="Sentiment Trend Analysis",
            template=self.theme,
            height=600
        )
        
        return fig
    
    def create_sentiment_summary_dashboard(self, df: pd.DataFrame,
                                         speaker_column: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive sentiment summary dashboard.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Optional column name for speaker
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Timeline', 'Sentiment Distribution', 
                          'Speaker Comparison', 'Moving Average'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Timeline
        timeline_fig = self.create_sentiment_timeline(df, speaker_column=speaker_column)
        for trace in timeline_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Distribution
        dist_fig = self.create_sentiment_distribution(df)
        for trace in dist_fig.data:
            fig.add_trace(trace, row=1, col=2)
        
        # Speaker comparison (if available)
        if speaker_column and speaker_column in df.columns:
            speaker_fig = self.create_speaker_sentiment_chart(df, speaker_column)
            for trace in speaker_fig.data:
                fig.add_trace(trace, row=2, col=1)
        
        # Moving average
        ma_fig = self.create_moving_average_chart(df)
        for trace in ma_fig.data:
            fig.add_trace(trace, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="MoodMeet Sentiment Analysis Dashboard",
            template=self.theme,
            height=800,
            showlegend=False
        )
        
        return fig


class StreamlitVisualizer:
    """Streamlit-specific visualization helpers."""
    
    def __init__(self):
        self.timeline_viz = MoodTimelineVisualizer()
    
    def display_sentiment_summary(self, df: pd.DataFrame, 
                                sentiment_summary: Dict) -> None:
        """
        Display sentiment summary in Streamlit.
        
        Args:
            df: DataFrame with sentiment data
            sentiment_summary: Dictionary with sentiment summary
        """
        if not sentiment_summary:
            st.warning("No sentiment data available.")
            return
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Messages",
                value=sentiment_summary.get('total_messages', 0)
            )
        
        with col2:
            avg_polarity = sentiment_summary.get('avg_polarity', 0)
            st.metric(
                label="Average Sentiment",
                value=f"{avg_polarity:.3f}",
                delta=f"{avg_polarity:.3f}"
            )
st.metric(
    label="Average Sentiment",
    value=f"{avg_polarity:.3f}"
)
        with col3:
            positive_ratio = sentiment_summary.get('positive_ratio', 0)
            st.metric(
                label="Positive Ratio",
                value=f"{positive_ratio:.1%}"
            )
        
        with col4:
            if 'sentiment_distribution' in sentiment_summary:
                sentiment_dist = sentiment_summary['sentiment_distribution']
                dominant_sentiment = max(sentiment_dist.items(), key=lambda x: x[1])[0]
                st.metric(
                    label="Dominant Sentiment",
                    value=dominant_sentiment.title()
                )
    
    def display_timeline_chart(self, df: pd.DataFrame,
                             speaker_column: Optional[str] = None) -> None:
        """
        Display timeline chart in Streamlit.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Optional column name for speaker
        """
        if df.empty:
            st.warning("No data available for timeline chart.")
            return
        
        fig = self.timeline_viz.create_sentiment_timeline(
            df, speaker_column=speaker_column
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_speaker_comparison(self, df: pd.DataFrame,
                                 speaker_column: str = 'speaker') -> None:
        """
        Display speaker comparison chart in Streamlit.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Column name for speaker
        """
        if df.empty or speaker_column not in df.columns:
            st.warning("No speaker data available.")
            return
        
        fig = self.timeline_viz.create_speaker_sentiment_chart(
            df, speaker_column=speaker_column
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def display_dashboard(self, df: pd.DataFrame,
                         sentiment_summary: Dict,
                         speaker_column: Optional[str] = None) -> None:
        """
        Display complete dashboard in Streamlit.
        
        Args:
            df: DataFrame with sentiment data
            sentiment_summary: Dictionary with sentiment summary
            speaker_column: Optional column name for speaker
        """
        # Summary metrics
        st.subheader("📊 Sentiment Summary")
        self.display_sentiment_summary(df, sentiment_summary)
        
        # Timeline chart
        st.subheader("📈 Sentiment Timeline")
        self.display_timeline_chart(df, speaker_column)
        
        # Speaker comparison
        if speaker_column and speaker_column in df.columns:
            st.subheader("👥 Speaker Sentiment Comparison")
            self.display_speaker_comparison(df, speaker_column)
        
        # Distribution and trend
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Distribution")
            dist_fig = self.timeline_viz.create_sentiment_distribution(df)
            st.plotly_chart(dist_fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Moving Average Trend")
            ma_fig = self.timeline_viz.create_moving_average_chart(df)
            st.plotly_chart(ma_fig, use_container_width=True)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        'text': [
            "We're falling behind schedule.",
            "Let's regroup and finish the draft today.",
            "I'm feeling a bit burned out.",
            "I think we can make it work if we focus.",
            "That sounds like a good plan."
        ],
        'speaker': ['Alice', 'Bob', 'Carol', 'David', 'Alice'],
        'polarity': [-0.2, 0.3, -0.4, 0.1, 0.5]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test visualizer
    viz = MoodTimelineVisualizer()
    
    # Create timeline
    timeline_fig = viz.create_sentiment_timeline(df, speaker_column='speaker')
    print("Timeline chart created successfully")
    
    # Create speaker comparison
    speaker_fig = viz.create_speaker_sentiment_chart(df, speaker_column='speaker')
    print("Speaker comparison chart created successfully")
    
    # Create distribution
    dist_fig = viz.create_sentiment_distribution(df)
    print("Distribution chart created successfully") 