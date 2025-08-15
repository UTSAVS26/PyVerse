import streamlit as st
import sys
import os
import time
import json
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.memory_game import MemoryGame, GameState
from ml.pattern_learner import PatternLearner, DifficultyLevel
from ml.adaptive_difficulty import AdaptiveDifficulty
from utils.timer import SessionTimer


class MemoAIApp:
    """Main Streamlit application for MemoAI."""
    
    def __init__(self):
        self.pattern_learner = PatternLearner()
        self.adaptive_difficulty = AdaptiveDifficulty(self.pattern_learner)
        self.session_timer = SessionTimer()
        self.current_game = None
        self.game_state = "waiting"
        self.revealed_cards = []
        self.matched_pairs = []
        
        # Initialize session state
        if 'game_sessions' not in st.session_state:
            st.session_state.game_sessions = []
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = 0
    
    def run(self):
        """Run the main application."""
        st.set_page_config(
            page_title="MemoAI - Adaptive Memory Game",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        self._load_custom_css()
        
        # Header
        st.title("🧠 MemoAI: Adaptive Memory Game Trainer")
        st.markdown("---")
        
        # Sidebar for controls
        self._render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_game_area()
        
        with col2:
            self._render_stats_panel()
        
        # Bottom section for insights
        self._render_insights_section()
    
    def _load_custom_css(self):
        """Load custom CSS for styling."""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .game-card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem;
            text-align: center;
            font-size: 2rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .game-card:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .matched-card {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border-color: #4CAF50;
        }
        .revealed-card {
            background: linear-gradient(45deg, #2196F3, #1976D2);
            color: white;
            border-color: #2196F3;
        }
        .stats-panel {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .insight-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar with game controls."""
        with st.sidebar:
            st.header("🎮 Game Controls")
            
            # Difficulty selection
            difficulty = st.selectbox(
                "Select Difficulty",
                ["Easy (3x3)", "Medium (4x4)", "Hard (5x5)", "Expert (6x6)"],
                index=1
            )
            
            # Extract grid size from difficulty
            grid_size_map = {"Easy (3x3)": 3, "Medium (4x4)": 4, "Hard (5x5)": 5, "Expert (6x6)": 6}
            grid_size = grid_size_map[difficulty]
            
            # Start/Reset game button
            if st.button("🔄 Start New Game", type="primary"):
                self._start_new_game(grid_size)
            
            st.markdown("---")
            
            # Game instructions
            st.subheader("📖 How to Play")
            st.markdown("""
            1. Click on cards to reveal them
            2. Find matching pairs to clear them
            3. Complete the game as quickly as possible
            4. The AI will analyze your performance
            """)
            
            st.markdown("---")
            
            # ML Model Status
            st.subheader("🧠 AI Status")
            if self.pattern_learner.is_trained:
                st.success("✅ ML Model Trained")
                st.info(f"Training sessions: {len(self.pattern_learner.load_data()['sessions'])}")
            else:
                st.warning("⚠️ ML Model Not Trained")
                st.info("Complete 5+ games to train the AI")
            
            # Train model button
            if st.button("🤖 Train AI Model"):
                with st.spinner("Training ML model..."):
                    result = self.pattern_learner.train_models()
                    if result["status"] == "success":
                        st.success("✅ Model trained successfully!")
                        st.json(result)
                    else:
                        st.error(f"❌ {result['message']}")
    
    def _start_new_game(self, grid_size: int):
        """Start a new game with the specified grid size."""
        self.current_game = MemoryGame(grid_size)
        self.current_game.start_game()
        self.game_state = "playing"
        self.revealed_cards = []
        self.matched_pairs = []
        
        # Start session timer
        self.session_timer.start_session(grid_size)
        
        # Update session state
        st.session_state.current_session_id += 1
        
        st.success(f"🎮 New game started! Grid size: {grid_size}x{grid_size}")
    
    def _render_game_area(self):
        """Render the main game area."""
        st.subheader("🎯 Memory Game")
        
        if self.current_game is None or self.game_state == "waiting":
            st.info("👆 Select a difficulty and click 'Start New Game' to begin!")
            return
        
        # Game board
        board_state = self.current_game.get_board_state()
        
        # Create grid layout
        cols = st.columns(self.current_game.grid_size)
        
        for row in range(self.current_game.grid_size):
            for col in range(self.current_game.grid_size):
                card_data = board_state[row][col]
                
                with cols[col]:
                    if card_data["is_hidden"]:
                        if st.button("❓", key=f"card_{row}_{col}", use_container_width=True):
                            self._handle_card_click(row, col)
                    elif card_data["is_revealed"]:
                        st.markdown(f"""
                        <div class="game-card revealed-card">
                            {card_data["value"]}
                        </div>
                        """, unsafe_allow_html=True)
                    elif card_data["is_matched"]:
                        st.markdown(f"""
                        <div class="game-card matched-card">
                            {card_data["value"]}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Game status
        if self.current_game:
            stats = self.current_game.get_game_stats()
            progress = stats["completion_percentage"]
            
            st.progress(progress / 100)
            st.caption(f"Progress: {stats['matched_pairs']}/{stats['total_pairs']} pairs found")
            
            # Check if game is complete
            if self.current_game.is_game_complete():
                self._handle_game_completion()
    
    def _handle_card_click(self, row: int, col: int):
        """Handle card click events."""
        if self.current_game is None:
            return
        
        result = self.current_game.click_card(row, col)
        
        if "error" in result:
            st.error(result["error"])
            return
        
        # Record move in session timer
        is_mistake = result.get("match_found", True) == False
        self.session_timer.record_move(is_mistake)
        
        # Handle match result
        if result.get("match_found"):
            st.success("🎉 Match found!")
        elif "match_found" in result and not result["match_found"]:
            st.warning("❌ No match. Try again!")
            # Schedule hiding cards after a delay
            time.sleep(1)
            self.current_game.hide_unmatched_cards()
    
    def _handle_game_completion(self):
        """Handle game completion."""
        if self.current_game is None:
            return
        
        # End session timer
        session_data = self.session_timer.end_session()
        
        # Get final game stats
        final_stats = self.current_game.get_game_stats()
        
        # Combine session data with game stats
        complete_session = {
            **session_data,
            **final_stats,
            "completed": True
        }
        
        # Add to pattern learner
        self.pattern_learner.add_session(complete_session)
        
        # Store in session state
        st.session_state.game_sessions.append(complete_session)
        
        # Show completion message
        st.balloons()
        st.success("🎉 Congratulations! You completed the game!")
        
        # Show final stats
        self._show_completion_stats(complete_session)
        
        # Update game state
        self.game_state = "completed"
    
    def _show_completion_stats(self, session_data: Dict[str, Any]):
        """Show completion statistics."""
        st.subheader("📊 Game Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⏱️ Total Time", f"{session_data['total_time']:.1f}s")
        
        with col2:
            st.metric("🎯 Moves", session_data['moves'])
        
        with col3:
            st.metric("❌ Mistakes", session_data['mistakes'])
        
        with col4:
            st.metric("⚡ Avg Reaction", f"{session_data['avg_reaction_time']:.2f}s")
    
    def _render_stats_panel(self):
        """Render the statistics panel."""
        st.subheader("📈 Live Statistics")
        
        if self.current_game is None:
            st.info("Start a game to see statistics!")
            return
        
        stats = self.current_game.get_game_stats()
        
        # Current game stats
        st.metric("⏱️ Time Elapsed", f"{stats['total_time']:.1f}s")
        st.metric("🎯 Moves Made", stats['moves'])
        st.metric("❌ Mistakes", stats['mistakes'])
        st.metric("⚡ Avg Reaction", f"{stats['avg_reaction_time']:.2f}s")
        
        # Progress bar
        progress = stats['completion_percentage']
        st.progress(progress / 100)
        st.caption(f"Completion: {progress:.1f}%")
        
        # Adaptive hints
        if self.current_game:
            hints = self.adaptive_difficulty.get_adaptive_hints(stats)
            st.subheader("💡 AI Hints")
            for hint in hints:
                st.info(hint)
    
    def _render_insights_section(self):
        """Render the insights and analytics section."""
        st.markdown("---")
        st.subheader("🧠 AI Insights & Analytics")
        
        if not st.session_state.game_sessions:
            st.info("Complete some games to see AI insights!")
            return
        
        # Create tabs for different insights
        tab1, tab2, tab3 = st.tabs(["📊 Performance Trends", "🎯 Personalized Feedback", "🤖 ML Predictions"])
        
        with tab1:
            self._render_performance_trends()
        
        with tab2:
            self._render_personalized_feedback()
        
        with tab3:
            self._render_ml_predictions()
    
    def _render_performance_trends(self):
        """Render performance trends visualization."""
        sessions = st.session_state.game_sessions
        
        if len(sessions) < 2:
            st.info("Complete at least 2 games to see trends!")
            return
        
        # Create DataFrame for plotting
        df = pd.DataFrame(sessions)
        df['session_number'] = range(1, len(df) + 1)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Reaction Time Trend', 'Mistakes Trend', 'Completion Time Trend', 'Performance Score Trend'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Reaction time trend
        fig.add_trace(
            go.Scatter(x=df['session_number'], y=df['avg_reaction_time'], 
                      mode='lines+markers', name='Reaction Time'),
            row=1, col=1
        )
        
        # Mistakes trend
        fig.add_trace(
            go.Scatter(x=df['session_number'], y=df['mistakes'], 
                      mode='lines+markers', name='Mistakes'),
            row=1, col=2
        )
        
        # Completion time trend
        fig.add_trace(
            go.Scatter(x=df['session_number'], y=df['total_time'], 
                      mode='lines+markers', name='Completion Time'),
            row=2, col=1
        )
        
        # Performance score trend (calculated)
        performance_scores = []
        for _, row in df.iterrows():
            score = self.adaptive_difficulty._evaluate_current_performance(row.to_dict())["overall_score"]
            performance_scores.append(score)
        
        fig.add_trace(
            go.Scatter(x=df['session_number'], y=performance_scores, 
                      mode='lines+markers', name='Performance Score'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_personalized_feedback(self):
        """Render personalized feedback."""
        if not st.session_state.game_sessions:
            return
        
        latest_session = st.session_state.game_sessions[-1]
        feedback = self.adaptive_difficulty.get_personalized_feedback(latest_session)
        
        # Performance summary
        st.subheader("📊 Performance Summary")
        performance = feedback["performance_summary"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{performance['overall_score']:.2f}")
        with col2:
            st.metric("Performance Level", performance['performance_level'].title())
        with col3:
            st.metric("Completion %", f"{performance['completion_percentage']:.1f}%")
        
        # Strengths
        if feedback["strengths"]:
            st.subheader("✅ Your Strengths")
            for strength in feedback["strengths"]:
                st.success(f"• {strength}")
        
        # Areas for improvement
        if feedback["areas_for_improvement"]:
            st.subheader("🎯 Areas for Improvement")
            for area in feedback["areas_for_improvement"]:
                st.warning(f"• {area}")
        
        # Next session recommendation
        st.subheader("🎮 Next Session Recommendation")
        recommendation = feedback["next_session_recommendation"]
        
        st.info(f"**Recommended Difficulty:** {recommendation['recommended_difficulty'].title()}")
        st.info(f"**Confidence:** {recommendation['confidence']:.2f}")
        st.info(f"**Reasoning:** {recommendation['reasoning']}")
        
        # Motivational message
        st.subheader("💪 Motivation")
        st.markdown(f"**{feedback['motivational_message']}**")
    
    def _render_ml_predictions(self):
        """Render ML predictions and model insights."""
        if not self.pattern_learner.is_trained:
            st.warning("ML model needs to be trained first!")
            return
        
        if not st.session_state.game_sessions:
            return
        
        latest_session = st.session_state.game_sessions[-1]
        
        # Difficulty prediction
        st.subheader("🎯 Difficulty Prediction")
        difficulty_pred = self.pattern_learner.predict_difficulty(latest_session)
        
        if "error" not in difficulty_pred:
            st.success(f"**Recommended Difficulty:** {difficulty_pred['recommended_difficulty'].title()}")
            st.info(f"**Confidence:** {difficulty_pred['confidence']:.2f}")
        
        # Performance prediction
        st.subheader("📈 Performance Prediction")
        performance_pred = self.pattern_learner.predict_performance(latest_session)
        
        if "error" not in performance_pred:
            st.info(f"**Predicted Completion Time:** {performance_pred['predicted_completion_time']:.1f}s")
        
        # Player insights
        st.subheader("🧠 Player Insights")
        insights = self.pattern_learner.get_player_insights(st.session_state.game_sessions)
        
        if "error" not in insights:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Sessions", insights["total_sessions"])
                st.metric("Avg Reaction Time", f"{insights['avg_reaction_time']:.2f}s")
                st.metric("Avg Mistakes", f"{insights['avg_mistakes']:.1f}")
            
            with col2:
                st.metric("Avg Completion Time", f"{insights['avg_completion_time']:.1f}s")
                st.metric("Improvement Trend", insights["improvement_trend"].title())
            
            # Strengths and weaknesses
            if insights["strengths"]:
                st.success("**Strengths:** " + ", ".join(insights["strengths"]))
            
            if insights["weaknesses"]:
                st.warning("**Areas to Work On:** " + ", ".join(insights["weaknesses"]))


def main():
    """Main function to run the Streamlit app."""
    app = MemoAIApp()
    app.run()


if __name__ == "__main__":
    main() 