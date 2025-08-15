"""
Mood Clustering Module for MoodMeet

Provides topic clustering and mood grouping using various algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import umap
import hdbscan
from collections import Counter


@dataclass
class ClusterResult:
    """Represents a clustering result."""
    cluster_id: int
    texts: List[str]
    centroid_text: str
    keywords: List[str]
    sentiment_avg: float
    size: int


class MoodClusterer:
    """Clusters text data based on content and sentiment."""
    
    def __init__(self, method: str = "kmeans", n_clusters: int = 3):
        """
        Initialize mood clusterer.
        
        Args:
            method: Clustering method ('kmeans', 'lda', 'umap_hdbscan')
            n_clusters: Number of clusters (for K-Means and LDA)
        """
        self.method = method
        self.n_clusters = n_clusters
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        
    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """Clean and prepare texts for clustering."""
        cleaned_texts = []
        for text in texts:
            # Basic cleaning
            cleaned = text.lower().strip()
            # Remove extra whitespace
            cleaned = ' '.join(cleaned.split())
            if cleaned:
                cleaned_texts.append(cleaned)
        return cleaned_texts
    
    def _extract_keywords(self, texts: List[str], cluster_indices: List[int], 
                         cluster_id: int) -> List[str]:
        """Extract keywords for a cluster."""
        cluster_texts = [texts[i] for i in range(len(texts)) 
                       if cluster_indices[i] == cluster_id]
        
        if not cluster_texts:
            return []
        
        # Simple keyword extraction based on frequency
        all_words = []
        for text in cluster_texts:
            words = text.split()
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                     'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
                     'them', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were'}
        
        # Get top keywords
        keywords = [(word, count) for word, count in word_counts.items() 
                   if word not in stop_words and len(word) > 2]
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, count in keywords[:10]]
    
    def _get_centroid_text(self, texts: List[str], cluster_indices: List[int], 
                          cluster_id: int) -> str:
        """Get representative text for a cluster."""
        cluster_texts = [texts[i] for i in range(len(texts)) 
                       if cluster_indices[i] == cluster_id]
        
        if not cluster_texts:
            return ""
        
        # Return the shortest text as centroid (often most concise)
        return min(cluster_texts, key=len)
    
    def fit_kmeans(self, texts: List[str]) -> Tuple[List[int], np.ndarray]:
        """Fit K-Means clustering."""
        # Vectorize texts
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Changed from 2 to 1 to allow single occurrences
            max_df=0.9  # Allow more common terms
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit K-Means
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.model.fit_predict(tfidf_matrix)
        
        return cluster_labels, tfidf_matrix.toarray()
    
    def fit_lda(self, texts: List[str]) -> Tuple[List[int], np.ndarray]:
        """Fit LDA topic modeling."""
        # Vectorize texts
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Changed from 2 to 1 to allow single occurrences
            max_df=0.9  # Allow more common terms
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit LDA
        self.model = LatentDirichletAllocation(
            n_components=self.n_clusters,
            random_state=42,
            max_iter=20
        )
        
        # Get topic distributions
        topic_distributions = self.model.fit_transform(tfidf_matrix)
        
        # Assign cluster labels based on dominant topic
        cluster_labels = np.argmax(topic_distributions, axis=1)
        
        return cluster_labels.tolist(), topic_distributions
    
    def fit_umap_hdbscan(self, texts: List[str]) -> Tuple[List[int], np.ndarray]:
        """Fit UMAP + HDBSCAN clustering."""
        # Vectorize texts
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Changed from 2 to 1 to allow single occurrences
            max_df=0.9  # Allow more common terms
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Reduce dimensionality with UMAP
        umap_reducer = umap.UMAP(
            n_neighbors=min(15, len(texts) - 1),
            n_components=50,
            random_state=42
        )
        
        umap_embeddings = umap_reducer.fit_transform(tfidf_matrix)
        
        # Cluster with HDBSCAN
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean'
        )
        
        cluster_labels = self.model.fit_predict(umap_embeddings)
        
        return cluster_labels.tolist(), umap_embeddings
    
    def fit(self, texts: List[str], sentiments: Optional[List[float]] = None) -> List[ClusterResult]:
        """
        Fit clustering model to texts.
        
        Args:
            texts: List of texts to cluster
            sentiments: Optional list of sentiment scores
            
        Returns:
            List of ClusterResult objects
        """
        # Clean texts
        cleaned_texts = self._prepare_texts(texts)
        
        if len(cleaned_texts) < 2:
            return []
        
        # Fit clustering model
        if self.method == "kmeans":
            cluster_labels, embeddings = self.fit_kmeans(cleaned_texts)
        elif self.method == "lda":
            cluster_labels, embeddings = self.fit_lda(cleaned_texts)
        elif self.method == "umap_hdbscan":
            cluster_labels, embeddings = self.fit_umap_hdbscan(cleaned_texts)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Create cluster results
        cluster_results = []
        unique_clusters = set(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise points in HDBSCAN
                continue
                
            # Get texts in this cluster
            cluster_texts = [texts[i] for i in range(len(texts)) 
                           if cluster_labels[i] == cluster_id]
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_texts, cluster_labels, cluster_id)
            
            # Get centroid text
            centroid_text = self._get_centroid_text(texts, cluster_labels, cluster_id)
            
            # Calculate average sentiment
            sentiment_avg = 0.0
            if sentiments:
                cluster_sentiments = [sentiments[i] for i in range(len(sentiments)) 
                                   if cluster_labels[i] == cluster_id]
                if cluster_sentiments:
                    sentiment_avg = np.mean(cluster_sentiments)
            
            result = ClusterResult(
                cluster_id=cluster_id,
                texts=cluster_texts,
                centroid_text=centroid_text,
                keywords=keywords,
                sentiment_avg=sentiment_avg,
                size=len(cluster_texts)
            )
            
            cluster_results.append(result)
        
        return cluster_results
    
    def get_clustering_summary(self, cluster_results: List[ClusterResult]) -> Dict:
        """
        Get summary of clustering results.
        
        Args:
            cluster_results: List of ClusterResult objects
            
        Returns:
            Dictionary with clustering summary
        """
        if not cluster_results:
            return {}
        
        summary = {
            'total_clusters': len(cluster_results),
            'total_texts': sum(cluster.size for cluster in cluster_results),
            'avg_cluster_size': np.mean([cluster.size for cluster in cluster_results]),
            'clusters': []
        }
        
        for cluster in cluster_results:
            cluster_info = {
                'cluster_id': cluster.cluster_id,
                'size': cluster.size,
                'sentiment_avg': cluster.sentiment_avg,
                'keywords': cluster.keywords[:5],  # Top 5 keywords
                'centroid_text': cluster.centroid_text
            }
            summary['clusters'].append(cluster_info)
        
        # Sort clusters by size
        summary['clusters'].sort(key=lambda x: x['size'], reverse=True)
        
        return summary


class TopicAnalyzer:
    """Analyzes topics and themes in text data."""
    
    def __init__(self):
        self.vectorizer = None
    
    def extract_topics(self, texts: List[str], n_topics: int = 5) -> List[Dict]:
        """
        Extract main topics from texts.
        
        Args:
            texts: List of texts to analyze
            n_topics: Number of topics to extract
            
        Returns:
            List of topic dictionaries
        """
        if len(texts) < 2:
            return []
        
        # Vectorize texts
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Use LDA for topic extraction
        lda = LatentDirichletAllocation(
            n_components=min(n_topics, len(texts)),
            random_state=42,
            max_iter=20
        )
        
        lda.fit(tfidf_matrix)
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            # Get top words for this topic
            top_word_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            
            topic_info = {
                'topic_id': topic_idx,
                'keywords': top_words,
                'weight': topic.sum()  # Topic weight
            }
            
            topics.append(topic_info)
        
        return topics
    
    def get_topic_summary(self, topics: List[Dict]) -> Dict:
        """
        Get summary of topic analysis.
        
        Args:
            topics: List of topic dictionaries
            
        Returns:
            Dictionary with topic summary
        """
        if not topics:
            return {}
        
        summary = {
            'total_topics': len(topics),
            'topics': []
        }
        
        for topic in topics:
            topic_summary = {
                'topic_id': topic['topic_id'],
                'main_keywords': topic['keywords'][:5],  # Top 5 keywords
                'weight': topic['weight']
            }
            summary['topics'].append(topic_summary)
        
        # Sort by weight
        summary['topics'].sort(key=lambda x: x['weight'], reverse=True)
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Test clustering
    texts = [
        "We're falling behind schedule.",
        "Let's regroup and finish the draft today.",
        "I'm feeling a bit burned out.",
        "I think we can make it work if we focus.",
        "That sounds like a good plan.",
        "The deadline is approaching fast.",
        "We need to prioritize our tasks.",
        "I'm confident we can deliver on time.",
        "Let's break this down into smaller tasks.",
        "The team is working well together."
    ]
    
    sentiments = [-0.2, 0.3, -0.4, 0.1, 0.5, -0.3, 0.2, 0.4, 0.1, 0.6]
    
    # Test K-Means clustering
    clusterer = MoodClusterer(method="kmeans", n_clusters=3)
    results = clusterer.fit(texts, sentiments)
    
    print("K-Means Clustering Results:")
    for result in results:
        print(f"Cluster {result.cluster_id}: {result.size} texts, "
              f"sentiment: {result.sentiment_avg:.3f}")
        print(f"Keywords: {result.keywords[:5]}")
        print(f"Centroid: {result.centroid_text}")
        print()
    
    # Test topic analysis
    topic_analyzer = TopicAnalyzer()
    topics = topic_analyzer.extract_topics(texts, n_topics=3)
    
    print("Topic Analysis Results:")
    for topic in topics:
        print(f"Topic {topic['topic_id']}: {topic['keywords'][:5]}")
        print() 