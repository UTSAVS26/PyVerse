import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import streamlit as st
import pandas as pd
import numpy as np

class ResultFormatter:
    """Formats and visualizes personality classification results."""
    
    def __init__(self):
        """Initialize the result formatter."""
        self.mbti_colors = {
            "INTJ": "#FF6B6B", "INTP": "#4ECDC4", "ENTJ": "#45B7D1", "ENTP": "#96CEB4",
            "INFJ": "#FFEAA7", "INFP": "#DDA0DD", "ENFJ": "#98D8C8", "ENFP": "#F7DC6F",
            "ISTJ": "#BB8FCE", "ISFJ": "#85C1E9", "ESTJ": "#F8C471", "ESFJ": "#82E0AA",
            "ISTP": "#F1948A", "ISFP": "#85C1E9", "ESTP": "#F8C471", "ESFP": "#82E0AA"
        }
    
    def format_personality_result(self, classification_result: Dict, personality_desc: Dict) -> str:
        """Format the personality result into a readable string."""
        mbti_type = classification_result.get("mbti_type", "UNKNOWN")
        confidence = classification_result.get("confidence", 0.0)
        method = classification_result.get("method", "unknown")
        
        # Get personality description
        name = personality_desc.get("name", "Unknown Type")
        strengths = personality_desc.get("strengths", [])
        careers = personality_desc.get("careers", [])
        description = personality_desc.get("description", "No description available.")
        
        # Format the result
        result_text = f"""
🧠 **Your MBTI Type: {mbti_type} ({name})**

**Confidence:** {confidence:.1%} (via {method} classification)

**Description:** {description}

**Strengths:** {', '.join(strengths)}

**Possible Careers:** {', '.join(careers)}

**Analysis Method:** {method.title()}
"""
        
        return result_text
    
    def create_radar_chart(self, dimension_scores: Dict) -> go.Figure:
        """Create a radar chart showing dimension scores."""
        if not dimension_scores:
            return None
        
        # Prepare data for radar chart
        categories = []
        values = []
        
        dimension_names = {
            "introversion_extraversion": "I/E",
            "sensing_intuition": "S/N", 
            "thinking_feeling": "T/F",
            "judging_perceiving": "J/P"
        }
        
        for dimension, scores in dimension_scores.items():
            if isinstance(scores, dict) and len(scores) == 2:
                # Calculate percentage for each dimension
                total = sum(scores.values())
                if total > 0:
                    # Get the dominant preference
                    dominant = max(scores, key=scores.get)
                    percentage = (scores[dominant] / total) * 100
                    
                    categories.append(dimension_names.get(dimension, dimension))
                    values.append(percentage)
        
        if not categories:
            return None
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Profile',
            line_color='rgb(32, 201, 151)',
            fillcolor='rgba(32, 201, 151, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="MBTI Dimension Profile",
            font=dict(size=12)
        )
        
        return fig
    
    def create_dimension_bar_chart(self, dimension_scores: Dict) -> go.Figure:
        """Create a bar chart showing dimension comparisons."""
        if not dimension_scores:
            return None
        
        # Prepare data
        dimensions = []
        i_scores = []
        e_scores = []
        s_scores = []
        n_scores = []
        t_scores = []
        f_scores = []
        j_scores = []
        p_scores = []
        
        for dimension, scores in dimension_scores.items():
            if isinstance(scores, dict):
                dimensions.append(dimension.replace("_", " ").title())
                
                if "I" in scores and "E" in scores:
                    i_scores.append(scores.get("I", 0))
                    e_scores.append(scores.get("E", 0))
                else:
                    i_scores.append(0)
                    e_scores.append(0)
                
                if "S" in scores and "N" in scores:
                    s_scores.append(scores.get("S", 0))
                    n_scores.append(scores.get("N", 0))
                else:
                    s_scores.append(0)
                    n_scores.append(0)
                
                if "T" in scores and "F" in scores:
                    t_scores.append(scores.get("T", 0))
                    f_scores.append(scores.get("F", 0))
                else:
                    t_scores.append(0)
                    f_scores.append(0)
                
                if "J" in scores and "P" in scores:
                    j_scores.append(scores.get("J", 0))
                    p_scores.append(scores.get("P", 0))
                else:
                    j_scores.append(0)
                    p_scores.append(0)
        
        # Create bar chart
        fig = go.Figure()
        
        # Add traces for each dimension pair
        if any(i_scores) or any(e_scores):
            fig.add_trace(go.Bar(name='I', x=dimensions, y=i_scores, marker_color='#FF6B6B'))
            fig.add_trace(go.Bar(name='E', x=dimensions, y=e_scores, marker_color='#4ECDC4'))
        
        if any(s_scores) or any(n_scores):
            fig.add_trace(go.Bar(name='S', x=dimensions, y=s_scores, marker_color='#45B7D1'))
            fig.add_trace(go.Bar(name='N', x=dimensions, y=n_scores, marker_color='#96CEB4'))
        
        if any(t_scores) or any(f_scores):
            fig.add_trace(go.Bar(name='T', x=dimensions, y=t_scores, marker_color='#FFEAA7'))
            fig.add_trace(go.Bar(name='F', x=dimensions, y=f_scores, marker_color='#DDA0DD'))
        
        if any(j_scores) or any(p_scores):
            fig.add_trace(go.Bar(name='J', x=dimensions, y=j_scores, marker_color='#98D8C8'))
            fig.add_trace(go.Bar(name='P', x=dimensions, y=p_scores, marker_color='#F7DC6F'))
        
        fig.update_layout(
            title="MBTI Dimension Scores",
            xaxis_title="Dimensions",
            yaxis_title="Score",
            barmode='group',
            font=dict(size=12)
        )
        
        return fig
    
    def create_confidence_gauge(self, confidence: float) -> go.Figure:
        """Create a gauge chart showing classification confidence."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Classification Confidence"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            title="Classification Confidence",
            font=dict(size=12)
        )
        
        return fig
    
    def create_mbti_distribution_chart(self, all_scores: Dict) -> go.Figure:
        """Create a bar chart showing MBTI type distribution scores."""
        if not all_scores:
            return None
        
        # Sort by score
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 8 types
        top_scores = sorted_scores[:8]
        
        types = [score[0] for score in top_scores]
        scores = [score[1] for score in top_scores]
        colors = [self.mbti_colors.get(mbti_type, "#CCCCCC") for mbti_type in types]
        
        fig = go.Figure(data=[
            go.Bar(
                x=types,
                y=scores,
                marker_color=colors,
                text=[f"{score:.3f}" for score in scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Top MBTI Type Predictions",
            xaxis_title="MBTI Type",
            yaxis_title="Confidence Score",
            font=dict(size=12)
        )
        
        return fig
    
    def format_detailed_analysis(self, analysis: Dict) -> str:
        """Format detailed analysis information."""
        if not analysis:
            return "No detailed analysis available."
        
        text_length = analysis.get("text_length", 0)
        word_count = analysis.get("word_count", 0)
        sentence_count = analysis.get("sentence_count", 0)
        sentiment = analysis.get("sentiment", {})
        
        analysis_text = f"""
**Text Analysis:**
- Characters: {text_length}
- Words: {word_count}
- Sentences: {sentence_count}

**Sentiment Analysis:**
- Positive: {sentiment.get('pos', 0):.3f}
- Negative: {sentiment.get('neg', 0):.3f}
- Neutral: {sentiment.get('neu', 0):.3f}
- Compound: {sentiment.get('compound', 0):.3f}
"""
        
        # Add dimension analysis if available
        dimension_analysis = analysis.get("dimension_analysis", {})
        if dimension_analysis:
            analysis_text += "\n**Dimension Analysis:**\n"
            for dimension, data in dimension_analysis.items():
                preference = data.get("preference", "Unknown")
                percentages = data.get("percentages", {})
                analysis_text += f"- {dimension.replace('_', ' ').title()}: {preference} "
                if percentages:
                    analysis_text += f"({max(percentages.values()):.1f}%)\n"
                else:
                    analysis_text += "\n"
        
        return analysis_text
    
    def display_streamlit_results(self, classification_result: Dict, personality_desc: Dict):
        """Display results in Streamlit format."""
        # Main result
        st.markdown(self.format_personality_result(classification_result, personality_desc))
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart
            if "dimensions" in classification_result:
                radar_fig = self.create_radar_chart(classification_result["dimensions"])
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
            
            # Confidence gauge
            confidence = classification_result.get("confidence", 0.0)
            gauge_fig = self.create_confidence_gauge(confidence)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            # Dimension bar chart
            if "dimensions" in classification_result:
                bar_fig = self.create_dimension_bar_chart(classification_result["dimensions"])
                if bar_fig:
                    st.plotly_chart(bar_fig, use_container_width=True)
            
            # MBTI distribution chart
            if "all_scores" in classification_result:
                dist_fig = self.create_mbti_distribution_chart(classification_result["all_scores"])
                if dist_fig:
                    st.plotly_chart(dist_fig, use_container_width=True)
        
        # Detailed analysis
        if "analysis" in classification_result:
            with st.expander("Detailed Analysis"):
                st.markdown(self.format_detailed_analysis(classification_result["analysis"]))
    
    def export_results(self, classification_result: Dict, personality_desc: Dict, filename: str = "personality_results.txt"):
        """Export results to a text file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("TextPersona - Personality Analysis Results\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(self.format_personality_result(classification_result, personality_desc))
                f.write("\n")
                
                if "analysis" in classification_result:
                    f.write(self.format_detailed_analysis(classification_result["analysis"]))
                
            print(f"✅ Results exported to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
            return False 