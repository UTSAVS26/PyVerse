"""
MoodMeet Demo Script

Demonstrates the complete MoodMeet functionality with sample data.
"""

import sys
import os
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data.uploader import TextInputHandler
from analysis.sentiment_analyzer import SentimentAnalyzer, SentimentTrendAnalyzer
from analysis.mood_clustering import MoodClusterer, TopicAnalyzer
from analysis.keyword_extractor import KeywordExtractor, PhraseExtractor
from visualization.mood_timeline import MoodTimelineVisualizer
from visualization.heatmap_generator import HeatmapGenerator


def main():
    """Run the MoodMeet demo."""
    
    print("🎯 MoodMeet - AI-Powered Meeting Mood Analyzer")
    print("=" * 60)
    
    # Sample meeting transcript
    sample_text = """
Alice: Good morning everyone, let's start our weekly team meeting.
Bob: Morning Alice, I'm ready to discuss the project updates.
Carol: Hi team, I have some concerns about our timeline.
David: Hello everyone, I think we're making good progress overall.
Alice: Let's begin with the project status. Bob, how's the frontend development going?
Bob: The frontend is coming along well. We've completed 70% of the user interface.
Carol: That's good, but I'm worried about the backend integration. We're falling behind schedule.
David: I understand your concern, Carol. The backend team is working overtime to catch up.
Alice: Let's regroup and finish the integration by Friday. Can we make that deadline?
Bob: I think we can make it work if we focus on the critical features first.
Carol: Honestly, I'm feeling a bit burned out with all the pressure.
David: I think we can make it work if we support each other better.
Alice: That sounds like a good plan. Let's break this down into smaller tasks.
Bob: We need to prioritize our tasks and communicate more effectively.
Carol: The deadline is approaching fast, but I'm confident we can deliver.
David: I'm confident we can deliver on time if we work together.
Alice: Let's break this down into smaller, manageable tasks.
Bob: The team is working well together, we just need better coordination.
Carol: I agree, let's focus on what we can control and support each other.
David: That's a great attitude, Carol. We'll get through this together.
Alice: Perfect! Let's end on a positive note. Great work everyone!
    """
    
    print("\n📝 Processing transcript...")
    
    try:
        # Initialize components
        text_handler = TextInputHandler()
        sentiment_analyzer = SentimentAnalyzer(model_type="vader")
        clusterer = MoodClusterer(method="kmeans", n_clusters=3)
        keyword_extractor = KeywordExtractor(method="tfidf")
        topic_analyzer = TopicAnalyzer()
        trend_analyzer = SentimentTrendAnalyzer()
        
        # Process text input
        print("1. Parsing transcript...")
        df, speaker_stats, is_valid, errors = text_handler.process_text_input(sample_text)
        
        if not is_valid:
            print("❌ Invalid transcript data:")
            for error in errors:
                print(f"   • {error}")
            return
        
        print(f"   ✅ Processed {len(df)} messages from {len(speaker_stats)} speakers")
        
        # Perform sentiment analysis
        print("\n2. Analyzing sentiment...")
        sentiment_results = sentiment_analyzer.analyze_dataframe(df)
        sentiment_summary = sentiment_analyzer.get_sentiment_summary(sentiment_results)
        
        print(f"   ✅ Average sentiment: {sentiment_summary.get('avg_polarity', 0):.3f}")
        print(f"   ✅ Positive ratio: {sentiment_summary.get('positive_ratio', 0):.1%}")
        
        # Perform clustering
        print("\n3. Clustering topics...")
        texts = df['text'].tolist()
        sentiments = sentiment_results['polarity'].tolist()
        cluster_results = clusterer.fit(texts, sentiments)
        clustering_summary = clusterer.get_clustering_summary(cluster_results)
        
        print(f"   ✅ Found {clustering_summary.get('total_clusters', 0)} topic clusters")
        
        # Extract keywords
        print("\n4. Extracting keywords...")
        keywords = keyword_extractor.extract_keywords(texts, max_keywords=10)
        keyword_summary = keyword_extractor.get_keyword_summary(keywords)
        
        print(f"   ✅ Extracted {len(keywords)} keywords")
        
        # Extract topics
        print("\n5. Analyzing topics...")
        topics = topic_analyzer.extract_topics(texts, n_topics=3)
        topic_summary = topic_analyzer.get_topic_summary(topics)
        
        print(f"   ✅ Identified {len(topics)} main topics")
        
        # Analyze trends
        print("\n6. Analyzing trends...")
        trend_df = trend_analyzer.analyze_trend(sentiment_results)
        trend_summary = trend_analyzer.get_trend_summary(trend_df)
        
        print(f"   ✅ Trend direction: {trend_summary.get('trend_direction', 'unknown')}")
        
        # Display results
        print("\n📊 Analysis Results")
        print("=" * 40)
        
        # Sentiment summary
        print("\n🎭 Sentiment Analysis:")
        print(f"   • Total messages: {sentiment_summary.get('total_messages', 0)}")
        print(f"   • Average sentiment: {sentiment_summary.get('avg_polarity', 0):.3f}")
        print(f"   • Positive ratio: {sentiment_summary.get('positive_ratio', 0):.1%}")
        
        if 'most_positive_text' in sentiment_summary:
            print(f"   • Most positive: \"{sentiment_summary['most_positive_text'][:50]}...\"")
        if 'most_negative_text' in sentiment_summary:
            print(f"   • Most negative: \"{sentiment_summary['most_negative_text'][:50]}...\"")
        
        # Clustering results
        print("\n🧠 Topic Clusters:")
        for cluster in cluster_results:
            print(f"   • Cluster {cluster.cluster_id}: {cluster.size} messages")
            print(f"     Keywords: {', '.join(cluster.keywords[:3])}")
            print(f"     Avg sentiment: {cluster.sentiment_avg:.3f}")
        
        # Keywords
        print("\n🔑 Top Keywords:")
        for i, keyword in enumerate(keywords[:5]):
            print(f"   {i+1}. {keyword.keyword} (score: {keyword.score:.3f})")
        
        # Topics
        print("\n📝 Main Topics:")
        for topic in topics:
            print(f"   • Topic {topic['topic_id']}: {', '.join(topic['keywords'][:3])}")
        
        # Trend
        print(f"\n📈 Trend Analysis:")
        print(f"   • Direction: {trend_summary.get('trend_direction', 'unknown')}")
        print(f"   • Magnitude: {trend_summary.get('trend_magnitude', 0):.3f}")
        print(f"   • Volatility: {trend_summary.get('volatility', 0):.3f}")
        
        # Speaker analysis
        print("\n👥 Speaker Analysis:")
        for speaker, stats in speaker_stats.items():
            print(f"   • {speaker}: {stats['message_count']} messages, "
                  f"{stats['total_words']} words")
        
        print("\n✅ Demo completed successfully!")
        print("\n🌐 To run the full web interface:")
        print("   streamlit run ui/streamlit_app.py")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 