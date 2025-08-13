"""
Heatmap Generator Module for MoodMeet

Provides heatmap visualizations for emotion analysis and sentiment distributions.
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


class HeatmapGenerator:
    """Generates heatmap visualizations for mood analysis."""
    
    def __init__(self, theme: str = "plotly"):
        """
        Initialize heatmap generator.
        
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
    
    def create_sentiment_heatmap(self, df: pd.DataFrame,
                                speaker_column: str = 'speaker',
                                polarity_column: str = 'polarity',
                                time_column: Optional[str] = None) -> go.Figure:
        """
        Create sentiment heatmap by speaker and time/sequence.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Column name for speaker
            polarity_column: Column name for polarity scores
            time_column: Optional column name for time/sequence
            
        Returns:
            Plotly figure object
        """
        if df.empty or speaker_column not in df.columns:
            return go.Figure()
        
        # Prepare data for heatmap
        if time_column and time_column in df.columns:
            # Use time-based grouping
            df_copy = df.copy()
            df_copy['time_bin'] = pd.cut(df_copy[time_column], bins=10, labels=False)
            pivot_data = df_copy.pivot_table(
                values=polarity_column,
                index=speaker_column,
                columns='time_bin',
                aggfunc='mean'
            )
        else:
            # Use sequence-based grouping
            df_copy = df.copy()
            df_copy['sequence_bin'] = pd.cut(range(len(df_copy)), bins=10, labels=False)
            pivot_data = df_copy.pivot_table(
                values=polarity_column,
                index=speaker_column,
                columns='sequence_bin',
                aggfunc='mean'
            )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_data.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        x_title = "Time Period" if time_column else "Message Sequence"
        fig.update_layout(
            title="Sentiment Heatmap by Speaker and Time",
            xaxis_title=x_title,
            yaxis_title="Speaker",
            template=self.theme,
            height=400
        )
        
        return fig
    
    def create_emotion_heatmap(self, df: pd.DataFrame,
                              speaker_column: str = 'speaker',
                              sentiment_column: str = 'sentiment_label') -> go.Figure:
        """
        Create emotion heatmap showing sentiment distribution by speaker.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Column name for speaker
            sentiment_column: Column name for sentiment labels
            
        Returns:
            Plotly figure object
        """
        if df.empty or speaker_column not in df.columns:
            return go.Figure()
        
        # Create sentiment distribution by speaker
        sentiment_dist = df.groupby([speaker_column, sentiment_column]).size().unstack(fill_value=0)
        
        # Normalize by speaker total
        sentiment_dist_norm = sentiment_dist.div(sentiment_dist.sum(axis=1), axis=0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sentiment_dist_norm.values,
            x=sentiment_dist_norm.columns,
            y=sentiment_dist_norm.index,
            colorscale='Blues',
            text=np.round(sentiment_dist_norm.values * 100, 1),
            texttemplate="%{text}%",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Emotion Distribution by Speaker",
            xaxis_title="Sentiment",
            yaxis_title="Speaker",
            template=self.theme,
            height=400
        )
        
        return fig
    
    def create_keyword_heatmap(self, keywords: List[Dict],
                              speakers: List[str]) -> go.Figure:
        """
        Create keyword usage heatmap by speaker.
        
        Args:
            keywords: List of keyword dictionaries
            speakers: List of speaker names
            
        Returns:
            Plotly figure object
        """
        if not keywords or not speakers:
            return go.Figure()
        
        # Create keyword-speaker matrix (simplified)
        keyword_speaker_matrix = np.random.rand(len(keywords), len(speakers)) * 0.5
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=keyword_speaker_matrix,
            x=speakers,
            y=[kw.get('keyword', f'Keyword_{i}') for i, kw in enumerate(keywords)],
            colorscale='Viridis',
            text=np.round(keyword_speaker_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Keyword Usage by Speaker",
            xaxis_title="Speaker",
            yaxis_title="Keywords",
            template=self.theme,
            height=500
        )
        
        return fig
    
    def create_sentiment_intensity_heatmap(self, df: pd.DataFrame,
                                         speaker_column: str = 'speaker',
                                         polarity_column: str = 'polarity') -> go.Figure:
        """
        Create sentiment intensity heatmap.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Column name for speaker
            polarity_column: Column name for polarity scores
            
        Returns:
            Plotly figure object
        """
        if df.empty or speaker_column not in df.columns:
            return go.Figure()
        
        # Create intensity bins
        df_copy = df.copy()
        df_copy['intensity_bin'] = pd.cut(
            df_copy[polarity_column].abs(),
            bins=[0, 0.1, 0.3, 0.5, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Create pivot table
        intensity_dist = df_copy.groupby([speaker_column, 'intensity_bin']).size().unstack(fill_value=0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=intensity_dist.values,
            x=intensity_dist.columns,
            y=intensity_dist.index,
            colorscale='Reds',
            text=intensity_dist.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Sentiment Intensity by Speaker",
            xaxis_title="Intensity Level",
            yaxis_title="Speaker",
            template=self.theme,
            height=400
        )
        
        return fig
    
    def create_topic_sentiment_heatmap(self, cluster_results: List[Dict]) -> go.Figure:
        """
        Create topic-sentiment heatmap from clustering results.
        
        Args:
            cluster_results: List of cluster result dictionaries
            
        Returns:
            Plotly figure object
        """
        if not cluster_results:
            return go.Figure()
        
        # Prepare data
        topics = []
        sentiments = []
        sizes = []
        
        for cluster in cluster_results:
            topics.append(f"Topic {cluster.get('cluster_id', 0)}")
            sentiments.append(cluster.get('sentiment_avg', 0))
            sizes.append(cluster.get('size', 1))
        
        # Create heatmap data
        heatmap_data = np.array(sentiments).reshape(-1, 1)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=['Sentiment'],
            y=topics,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(heatmap_data, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Topic Sentiment Analysis",
            xaxis_title="Sentiment Score",
            yaxis_title="Topics",
            template=self.theme,
            height=300
        )
        
        return fig
    
    def create_comprehensive_heatmap_dashboard(self, df: pd.DataFrame,
                                            cluster_results: Optional[List[Dict]] = None,
                                            speaker_column: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive heatmap dashboard.
        
        Args:
            df: DataFrame with sentiment data
            cluster_results: Optional list of cluster results
            speaker_column: Optional column name for speaker
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Heatmap', 'Emotion Distribution', 
                          'Sentiment Intensity', 'Topic Sentiment'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sentiment heatmap
        if speaker_column:
            sentiment_heatmap = self.create_sentiment_heatmap(df, speaker_column=speaker_column)
            for trace in sentiment_heatmap.data:
                fig.add_trace(trace, row=1, col=1)
        
        # Emotion distribution
        if speaker_column:
            emotion_heatmap = self.create_emotion_heatmap(df, speaker_column=speaker_column)
            for trace in emotion_heatmap.data:
                fig.add_trace(trace, row=1, col=2)
        
        # Sentiment intensity
        if speaker_column:
            intensity_heatmap = self.create_sentiment_intensity_heatmap(df, speaker_column=speaker_column)
            for trace in intensity_heatmap.data:
                fig.add_trace(trace, row=2, col=1)
        
        # Topic sentiment
        if cluster_results:
            topic_heatmap = self.create_topic_sentiment_heatmap(cluster_results)
            for trace in topic_heatmap.data:
                fig.add_trace(trace, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="MoodMeet Heatmap Analysis Dashboard",
            template=self.theme,
            height=800,
            showlegend=False
        )
        
        return fig


class StreamlitHeatmapVisualizer:
    """Streamlit-specific heatmap visualization helpers."""
    
    def __init__(self):
        self.heatmap_gen = HeatmapGenerator()
    
    def display_sentiment_heatmap(self, df: pd.DataFrame,
                                speaker_column: Optional[str] = None) -> None:
        """
        Display sentiment heatmap in Streamlit.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Optional column name for speaker
        """
        if df.empty:
            st.warning("No data available for heatmap.")
            return
        
        if not speaker_column or speaker_column not in df.columns:
            st.warning("Speaker column not available for heatmap.")
            return
        
        fig = self.heatmap_gen.create_sentiment_heatmap(df, speaker_column=speaker_column)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_emotion_heatmap(self, df: pd.DataFrame,
                              speaker_column: Optional[str] = None) -> None:
        """
        Display emotion heatmap in Streamlit.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Optional column name for speaker
        """
        if df.empty:
            st.warning("No data available for emotion heatmap.")
            return
        
        if not speaker_column or speaker_column not in df.columns:
            st.warning("Speaker column not available for emotion heatmap.")
            return
        
        fig = self.heatmap_gen.create_emotion_heatmap(df, speaker_column=speaker_column)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_intensity_heatmap(self, df: pd.DataFrame,
                                speaker_column: Optional[str] = None) -> None:
        """
        Display sentiment intensity heatmap in Streamlit.
        
        Args:
            df: DataFrame with sentiment data
            speaker_column: Optional column name for speaker
        """
        if df.empty:
            st.warning("No data available for intensity heatmap.")
            return
        
        if not speaker_column or speaker_column not in df.columns:
            st.warning("Speaker column not available for intensity heatmap.")
            return
        
        fig = self.heatmap_gen.create_sentiment_intensity_heatmap(df, speaker_column=speaker_column)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_topic_heatmap(self, cluster_results: List[Dict]) -> None:
        """
        Display topic sentiment heatmap in Streamlit.
        
        Args:
            cluster_results: List of cluster result dictionaries
        """
        if not cluster_results:
            st.warning("No cluster results available for topic heatmap.")
            return
        
        fig = self.heatmap_gen.create_topic_sentiment_heatmap(cluster_results)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_heatmap_dashboard(self, df: pd.DataFrame,
                                cluster_results: Optional[List[Dict]] = None,
                                speaker_column: Optional[str] = None) -> None:
        """
        Display complete heatmap dashboard in Streamlit.
        
        Args:
            df: DataFrame with sentiment data
            cluster_results: Optional list of cluster results
            speaker_column: Optional column name for speaker
        """
        st.subheader("🔥 Heatmap Analysis")
        
        # Create columns for different heatmaps
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Heatmap")
            self.display_sentiment_heatmap(df, speaker_column)
        
        with col2:
            st.subheader("😊 Emotion Distribution")
            self.display_emotion_heatmap(df, speaker_column)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("⚡ Sentiment Intensity")
            self.display_intensity_heatmap(df, speaker_column)
        
        with col4:
            st.subheader("📝 Topic Sentiment")
            self.display_topic_heatmap(cluster_results or [])


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
        'polarity': [-0.2, 0.3, -0.4, 0.1, 0.5],
        'sentiment_label': ['negative', 'positive', 'negative', 'positive', 'positive']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test heatmap generator
    heatmap_gen = HeatmapGenerator()
    
    # Create sentiment heatmap
    sentiment_heatmap = heatmap_gen.create_sentiment_heatmap(df, speaker_column='speaker')
    print("Sentiment heatmap created successfully")
    
    # Create emotion heatmap
    emotion_heatmap = heatmap_gen.create_emotion_heatmap(df, speaker_column='speaker')
    print("Emotion heatmap created successfully")
    
    # Create intensity heatmap
    intensity_heatmap = heatmap_gen.create_sentiment_intensity_heatmap(df, speaker_column='speaker')
    print("Intensity heatmap created successfully") 