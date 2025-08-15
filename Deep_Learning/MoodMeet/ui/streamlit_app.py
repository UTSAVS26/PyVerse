"""
MoodMeet Streamlit Application

Main web interface for the MoodMeet sentiment analysis tool.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.uploader import TextInputHandler
from analysis.sentiment_analyzer import SentimentAnalyzer, SentimentTrendAnalyzer
from analysis.mood_clustering import MoodClusterer, TopicAnalyzer
from analysis.keyword_extractor import KeywordExtractor, PhraseExtractor
from visualization.mood_timeline import StreamlitVisualizer
from visualization.heatmap_generator import StreamlitHeatmapVisualizer


# Page configuration
st.set_page_config(
    page_title="MoodMeet - AI-Powered Meeting Mood Analyzer",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📅 MoodMeet</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">AI-Powered Meeting Mood Analyzer</h2>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Sentiment Analysis Model")
        model_type = st.selectbox(
            "Choose sentiment analysis model:",
            ["vader", "textblob", "ensemble"],
            help="VADER: Rule-based, good for social media. TextBlob: Simple and effective. Ensemble: Combines multiple models."
        )
        
        # Clustering method
        st.subheader("Clustering Method")
        clustering_method = st.selectbox(
            "Choose clustering method:",
            ["kmeans", "lda", "umap_hdbscan"],
            help="K-Means: Traditional clustering. LDA: Topic modeling. UMAP+HDBSCAN: Advanced clustering."
        )
        
        n_clusters = st.slider("Number of clusters:", 2, 8, 3)
        
        # Keyword extraction method
        st.subheader("Keyword Extraction")
        keyword_method = st.selectbox(
            "Choose keyword extraction method:",
            ["tfidf", "rake", "yake", "ensemble"],
            help="TF-IDF: Term frequency analysis. RAKE: Rapid keyword extraction. YAKE: Yet another keyword extractor."
        )
        
        # Visualization settings
        st.subheader("Visualization Settings")
        chart_theme = st.selectbox("Chart theme:", ["plotly", "plotly_dark", "plotly_white"])
        
        # Example data
        st.subheader("📝 Example Data")
        if st.button("Load Example Transcript"):
            example_text = """
Alice: We're falling behind schedule.
Bob: Let's regroup and finish the draft today.
Carol: Honestly, I'm feeling a bit burned out.
David: I think we can make it work if we focus.
Alice: That sounds like a good plan.
Bob: We need to prioritize our tasks.
Carol: The deadline is approaching fast.
David: I'm confident we can deliver on time.
Alice: Let's break this down into smaller tasks.
Bob: The team is working well together.
            """
            st.session_state['input_text'] = example_text.strip()
            st.success("Example transcript loaded!")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Input", "📊 Analysis", "📈 Visualizations", "📋 Results"])
    
    with tab1:
        st.header("📝 Input Data")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "File Upload"],
            horizontal=True
        )
        
        if input_method == "Text Input":
            # Text input
            input_text = st.text_area(
                "Enter your meeting transcript:",
                value=st.session_state.get('input_text', ''),
                height=300,
                placeholder="Enter transcript in format:\nAlice: Hello everyone.\nBob: Hi Alice, how are you?\n..."
            )
            
            if st.button("Analyze Transcript", type="primary"):
                if input_text.strip():
                    st.session_state['input_text'] = input_text
                    process_transcript(input_text, model_type, clustering_method, n_clusters, keyword_method, chart_theme)
                else:
                    st.error("Please enter some text to analyze.")
        
        else:
            # File upload
            uploaded_file = st.file_uploader(
                "Upload transcript file:",
                type=['txt', 'csv'],
                help="Upload a text file or CSV with transcript data."
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        st.write("CSV file uploaded successfully!")
                        st.dataframe(df.head())
                    else:
                        content = uploaded_file.read().decode('utf-8')
                        st.text_area("File content:", content, height=200)
                        
                        if st.button("Analyze Uploaded File", type="primary"):
                            process_transcript(content, model_type, clustering_method, n_clusters, keyword_method, chart_theme)
                            
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    
    with tab2:
        st.header("📊 Analysis Results")
        
        if 'analysis_results' in st.session_state:
            display_analysis_results(st.session_state['analysis_results'])
        else:
            st.info("Please input transcript data and run analysis to see results here.")
    
    with tab3:
        st.header("📈 Visualizations")
        
        if 'analysis_results' in st.session_state:
            display_visualizations(st.session_state['analysis_results'], chart_theme)
        else:
            st.info("Please input transcript data and run analysis to see visualizations here.")
    
    with tab4:
        st.header("📋 Detailed Results")
        
        if 'analysis_results' in st.session_state:
            display_detailed_results(st.session_state['analysis_results'])
        else:
            st.info("Please input transcript data and run analysis to see detailed results here.")


def process_transcript(text: str, model_type: str, clustering_method: str, 
                     n_clusters: int, keyword_method: str, chart_theme: str):
    """Process transcript and perform analysis."""
    
    with st.spinner("Processing transcript..."):
        try:
            # Initialize components
            text_handler = TextInputHandler()
            sentiment_analyzer = SentimentAnalyzer(model_type=model_type)
            clusterer = MoodClusterer(method=clustering_method, n_clusters=n_clusters)
            keyword_extractor = KeywordExtractor(method=keyword_method)
            topic_analyzer = TopicAnalyzer()
            trend_analyzer = SentimentTrendAnalyzer()
            
            # Process text input
            df, speaker_stats, is_valid, errors = text_handler.process_text_input(text)
            
            if not is_valid:
                st.error("Invalid transcript data:")
                for error in errors:
                    st.error(f"• {error}")
                return
            
            # Perform sentiment analysis
            sentiment_results = sentiment_analyzer.analyze_dataframe(df)
            sentiment_summary = sentiment_analyzer.get_sentiment_summary(sentiment_results)
            
            # Merge results
            df_with_sentiment = pd.concat([df, sentiment_results], axis=1)
            
            # Perform clustering
            texts = df['text'].tolist()
            sentiments = sentiment_results['polarity'].tolist()
            cluster_results = clusterer.fit(texts, sentiments)
            clustering_summary = clusterer.get_clustering_summary(cluster_results)
            
            # Extract keywords
            keywords = keyword_extractor.extract_keywords(texts, max_keywords=15)
            keyword_summary = keyword_extractor.get_keyword_summary(keywords)
            
            # Extract topics
            topics = topic_analyzer.extract_topics(texts, n_topics=5)
            topic_summary = topic_analyzer.get_topic_summary(topics)
            
            # Analyze trends
            trend_df = trend_analyzer.analyze_trend(sentiment_results)
            trend_summary = trend_analyzer.get_trend_summary(trend_df)
            
            # Store results
            st.session_state['analysis_results'] = {
                'df': df_with_sentiment,
                'speaker_stats': speaker_stats,
                'sentiment_summary': sentiment_summary,
                'cluster_results': cluster_results,
                'clustering_summary': clustering_summary,
                'keywords': keywords,
                'keyword_summary': keyword_summary,
                'topics': topics,
                'topic_summary': topic_summary,
                'trend_df': trend_df,
                'trend_summary': trend_summary,
                'chart_theme': chart_theme
            }
            
            st.success("Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.exception(e)


def display_analysis_results(results: Dict):
    """Display analysis results."""
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Messages",
            results['sentiment_summary'].get('total_messages', 0)
        )
    
    with col2:
        avg_polarity = results['sentiment_summary'].get('avg_polarity', 0)
        st.metric(
            "Average Sentiment",
            f"{avg_polarity:.3f}",
            delta=f"{avg_polarity:.3f}"
        )
    
    with col3:
        positive_ratio = results['sentiment_summary'].get('positive_ratio', 0)
        st.metric(
            "Positive Ratio",
            f"{positive_ratio:.1%}"
        )
    
    with col4:
        if 'sentiment_distribution' in results['sentiment_summary']:
            sentiment_dist = results['sentiment_summary']['sentiment_distribution']
            dominant_sentiment = max(sentiment_dist.items(), key=lambda x: x[1])[0]
            st.metric(
                "Dominant Sentiment",
                dominant_sentiment.title()
            )
    
    # Sentiment summary
    st.subheader("📊 Sentiment Analysis Summary")
    
    if 'most_positive_text' in results['sentiment_summary']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Positive Statement:**")
            st.info(results['sentiment_summary']['most_positive_text'])
        
        with col2:
            st.markdown("**Most Negative Statement:**")
            st.warning(results['sentiment_summary']['most_negative_text'])
    
    # Clustering results
    if results['cluster_results']:
        st.subheader("🧠 Topic Clusters")
        
        for cluster in results['cluster_results']:
            with st.expander(f"Cluster {cluster.cluster_id} ({cluster.size} messages)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Representative text:** {cluster.centroid_text}")
                    st.markdown(f"**Average sentiment:** {cluster.sentiment_avg:.3f}")
                
                with col2:
                    st.markdown("**Keywords:**")
                    for keyword in cluster.keywords[:5]:
                        st.markdown(f"• {keyword}")
    
    # Keywords
    if results['keywords']:
        st.subheader("🔑 Key Topics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Keywords:**")
            for i, keyword in enumerate(results['keywords'][:10]):
                st.markdown(f"{i+1}. **{keyword.keyword}** (score: {keyword.score:.3f})")
        
        with col2:
            if 'keyword_categories' in results['keyword_summary']:
                st.markdown("**Keyword Categories:**")
                for category, keywords in results['keyword_summary']['keyword_categories'].items():
                    st.markdown(f"**{category.title()}:** {len(keywords)} keywords")


def display_visualizations(results: Dict, chart_theme: str):
    """Display visualizations."""
    
    # Initialize visualizers
    timeline_viz = StreamlitVisualizer()
    heatmap_viz = StreamlitHeatmapVisualizer()
    
    # Timeline and speaker comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Timeline")
        timeline_viz.display_timeline_chart(results['df'], speaker_column='speaker')
    
    with col2:
        st.subheader("👥 Speaker Comparison")
        timeline_viz.display_speaker_comparison(results['df'], speaker_column='speaker')
    
    # Distribution and trend
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Sentiment Distribution")
        dist_fig = timeline_viz.timeline_viz.create_sentiment_distribution(results['df'])
        st.plotly_chart(dist_fig, use_container_width=True)
    
    with col4:
        st.subheader("📈 Moving Average Trend")
        ma_fig = timeline_viz.timeline_viz.create_moving_average_chart(results['df'])
        st.plotly_chart(ma_fig, use_container_width=True)
    
    # Heatmaps
    st.subheader("🔥 Heatmap Analysis")
    heatmap_viz.display_heatmap_dashboard(
        results['df'],
        cluster_results=results['cluster_results'],
        speaker_column='speaker'
    )


def display_detailed_results(results: Dict):
    """Display detailed analysis results."""
    
    # Raw data
    st.subheader("📋 Raw Data")
    
    tab1, tab2, tab3 = st.tabs(["Processed Data", "Sentiment Results", "Speaker Statistics"])
    
    with tab1:
        st.dataframe(results['df'])
    
    with tab2:
        sentiment_cols = ['text', 'polarity', 'sentiment_label', 'confidence']
        sentiment_df = results['df'][sentiment_cols].copy()
        st.dataframe(sentiment_df)
    
    with tab3:
        speaker_stats_df = pd.DataFrame(results['speaker_stats']).T
        st.dataframe(speaker_stats_df)
    
    # Download options
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = results['df'].to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="moodmeet_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create summary report
        summary_text = f"""
MoodMeet Analysis Report

Summary:
- Total Messages: {results['sentiment_summary'].get('total_messages', 0)}
- Average Sentiment: {results['sentiment_summary'].get('avg_polarity', 0):.3f}
- Positive Ratio: {results['sentiment_summary'].get('positive_ratio', 0):.1%}

Sentiment Distribution:
{results['sentiment_summary'].get('sentiment_distribution', {})}

Top Keywords:
{[kw.keyword for kw in results['keywords'][:10]]}
        """
        st.download_button(
            label="Download Report",
            data=summary_text,
            file_name="moodmeet_report.txt",
            mime="text/plain"
        )
    
    with col3:
        st.info("💡 Tip: Use the CSV file for further analysis in Excel or other tools.")


if __name__ == "__main__":
    main() 