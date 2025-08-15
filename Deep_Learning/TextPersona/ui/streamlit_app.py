import streamlit as st
import sys
import os
import pandas as pd

# Add the core directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from prompt_interface import PromptInterface
from classifier_zero_shot import ZeroShotClassifier
from classifier_rules import RuleBasedClassifier
from result_formatter import ResultFormatter

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="TextPersona - Personality Predictor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧠 TextPersona</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Personality Type Predictor from Text Prompts</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_type = st.selectbox(
            "Classification Method",
            ["Zero-shot (Recommended)", "Rule-based", "Auto-select"],
            help="Choose the classification method. Zero-shot uses AI models, rule-based uses keyword analysis."
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum confidence level for classification"
        )
        
        # Show detailed analysis
        show_details = st.checkbox(
            "Show Detailed Analysis",
            value=True,
            help="Display detailed analysis and visualizations"
        )
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        TextPersona analyzes your responses to introspective questions to predict your MBTI personality type.
        
        **Features:**
        - 🤖 AI-powered classification
        - 📈 Interactive visualizations
        - 📝 Detailed analysis
        - 💾 Export results
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Take Quiz", "📊 Results", "ℹ️ About"])
    
    with tab1:
        st.header("🎯 Personality Assessment")
        st.markdown("Answer the following questions thoughtfully to discover your personality type.")
        
        # Initialize components
        if 'prompt_interface' not in st.session_state:
            st.session_state.prompt_interface = PromptInterface()
        
        if 'classifier' not in st.session_state:
            if model_type == "Zero-shot (Recommended)" or model_type == "Auto-select":
                try:
                    st.session_state.classifier = ZeroShotClassifier()
                except Exception as e:
                    st.warning(f"Zero-shot classifier failed to load: {e}")
                    st.session_state.classifier = RuleBasedClassifier()
            else:
                st.session_state.classifier = RuleBasedClassifier()
        
        if 'formatter' not in st.session_state:
            st.session_state.formatter = ResultFormatter()
        
        # Run the interface
        responses = st.session_state.prompt_interface.run_streamlit_interface()
        
        if responses:
            st.success("✅ All questions answered! Processing your responses...")
            
            # Format responses for classification
            formatted_text = st.session_state.prompt_interface.format_responses_for_classifier(responses)
            
            # Classify personality
            with st.spinner("Analyzing your personality..."):
                classification_result = st.session_state.classifier.classify_personality(formatted_text)
                
                # Get personality description
                personality_desc = st.session_state.classifier.get_personality_description(
                    classification_result["mbti_type"]
                )
                
                # Store results in session state
                st.session_state.classification_result = classification_result
                st.session_state.personality_desc = personality_desc
                st.session_state.responses = responses
                
                # Check confidence threshold
                confidence = classification_result.get("confidence", 0.0)
                if confidence < confidence_threshold:
                    st.warning(f"⚠️ Low confidence ({confidence:.1%}). Consider retaking the quiz with more detailed responses.")
                
                st.success("🎉 Analysis complete!")
                
                # Switch to results tab
                st.switch_page("📊 Results")
    
    with tab2:
        st.header("📊 Your Results")
        
        if 'classification_result' in st.session_state and 'personality_desc' in st.session_state:
            # Display results
            st.session_state.formatter.display_streamlit_results(
                st.session_state.classification_result,
                st.session_state.personality_desc
            )
            
            # Export option
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Export Results"):
                    success = st.session_state.formatter.export_results(
                        st.session_state.classification_result,
                        st.session_state.personality_desc
                    )
                    if success:
                        st.success("✅ Results exported to personality_results.txt")
                    else:
                        st.error("❌ Failed to export results")
            
            with col2:
                if st.button("🔄 Retake Quiz"):
                    # Clear session state
                    for key in ['classification_result', 'personality_desc', 'responses']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        else:
            st.info("👆 Take the quiz first to see your results!")
    
    with tab3:
        st.header("ℹ️ About TextPersona")
        
        st.markdown("""
        ### 🧠 What is TextPersona?
        
        TextPersona is an AI-powered personality assessment tool that predicts your MBTI (Myers-Briggs Type Indicator) 
        personality type based on your responses to introspective questions.
        
        ### 🔬 How it Works
        
        1. **Questionnaire**: Answer 10 carefully crafted questions about your preferences and behaviors
        2. **AI Analysis**: Our system analyzes your responses using advanced NLP techniques
        3. **Personality Prediction**: Get your predicted MBTI type with confidence scores
        4. **Detailed Insights**: Explore your personality traits, strengths, and career suggestions
        
        ### 🎯 MBTI Types
        
        The MBTI framework categorizes personalities into 16 types based on four dimensions:
        
        - **I/E**: Introversion vs Extraversion
        - **S/N**: Sensing vs Intuition  
        - **T/F**: Thinking vs Feeling
        - **J/P**: Judging vs Perceiving
        
        ### 🤖 Technology
        
        - **Zero-shot Classification**: Uses pre-trained AI models for accurate predictions
        - **Rule-based Fallback**: Keyword analysis when AI models are unavailable
        - **Interactive Visualizations**: Charts and graphs to understand your profile
        - **Privacy-focused**: No personal data is stored or shared
        
        ### 📈 Features
        
        - 🎯 **Accurate Predictions**: Advanced AI models trained on personality data
        - 📊 **Visual Analytics**: Interactive charts showing your personality profile
        - 💼 **Career Guidance**: Suggested careers based on your personality type
        - 📝 **Detailed Analysis**: Comprehensive breakdown of your responses
        - 💾 **Export Results**: Save your results for future reference
        
        ### 🔒 Privacy
        
        Your responses are processed locally and are not stored or shared. The analysis is completely anonymous.
        
        ### 📚 Learn More
        
        - [MBTI Foundation](https://www.myersbriggs.org/)
        - [16 Personality Types](https://www.16personalities.com/)
        - [Personality Psychology](https://en.wikipedia.org/wiki/Personality_psychology)
        """)
        
        # Show system info
        with st.expander("🔧 System Information"):
            st.code(f"""
            Python: {sys.version}
            Streamlit: {st.__version__}
            Pandas: {pd.__version__}
            """)

if __name__ == "__main__":
    main() 