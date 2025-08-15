import pytest
import sys
import os
import json
import time
from unittest.mock import Mock, patch
import numpy as np

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.memory_game import MemoryGame, Card, CardState, GameState
from utils.timer import GameTimer, SessionTimer
from ml.pattern_learner import PatternLearner, DifficultyLevel
from ml.adaptive_difficulty import AdaptiveDifficulty


class TestGameTimer:
    """Test cases for the GameTimer class."""
    
    def test_timer_initialization(self):
        """Test timer initialization."""
        timer = GameTimer()
        assert timer.start_time is None
        assert timer.end_time is None
        assert timer.move_times == []
        assert timer.last_move_time is None
    
    def test_start_game(self):
        """Test starting a game timer."""
        timer = GameTimer()
        timer.start_game()
        
        assert timer.start_time is not None
        assert timer.last_move_time is not None
        assert timer.move_times == []
    
    def test_record_move(self):
        """Test recording moves."""
        timer = GameTimer()
        timer.start_game()
        
        # Record first move
        reaction_time = timer.record_move()
        assert reaction_time >= 0
        assert len(timer.move_times) == 1
        
        # Record second move
        time.sleep(0.1)  # Small delay
        reaction_time2 = timer.record_move()
        assert reaction_time2 > 0
        assert len(timer.move_times) == 2
    
    def test_end_game(self):
        """Test ending a game."""
        timer = GameTimer()
        timer.start_game()
        time.sleep(0.1)
        
        total_time = timer.end_game()
        assert total_time > 0
        assert timer.end_time is not None
    
    def test_get_average_reaction_time(self):
        """Test calculating average reaction time."""
        timer = GameTimer()
        timer.start_game()
        
        # Record some moves
        timer.record_move()
        time.sleep(0.1)
        timer.record_move()
        time.sleep(0.1)
        timer.record_move()
        
        avg_time = timer.get_average_reaction_time()
        assert avg_time > 0
    
    def test_get_reaction_time_stats(self):
        """Test getting comprehensive reaction time statistics."""
        timer = GameTimer()
        timer.start_game()
        
        # Record moves
        timer.record_move()
        time.sleep(0.1)
        timer.record_move()
        time.sleep(0.1)
        timer.record_move()
        
        stats = timer.get_reaction_time_stats()
        assert 'avg_reaction_time' in stats
        assert 'min_reaction_time' in stats
        assert 'max_reaction_time' in stats
        assert 'std_reaction_time' in stats
        assert 'total_moves' in stats
        assert stats['total_moves'] == 3


class TestSessionTimer:
    """Test cases for the SessionTimer class."""
    
    def test_session_timer_initialization(self):
        """Test session timer initialization."""
        timer = SessionTimer()
        assert timer.sessions == []
        assert timer.current_session is None
    
    def test_start_session(self):
        """Test starting a session."""
        timer = SessionTimer()
        timer.start_session(4)
        
        assert timer.current_session is not None
        assert timer.current_session['grid_size'] == 4
        assert timer.current_session['session_id'] == 1
        assert timer.current_session['mistakes'] == 0
        assert timer.current_session['completed'] == False
    
    def test_record_move(self):
        """Test recording moves in a session."""
        timer = SessionTimer()
        timer.start_session(4)
        
        reaction_time = timer.record_move()
        assert reaction_time >= 0
        assert timer.current_session['mistakes'] == 0
        
        # Record a mistake
        reaction_time = timer.record_move(is_mistake=True)
        assert timer.current_session['mistakes'] == 1
    
    def test_end_session(self):
        """Test ending a session."""
        timer = SessionTimer()
        timer.start_session(4)
        
        # Record some moves
        timer.record_move()
        timer.record_move(is_mistake=True)
        timer.record_move()
        
        session_data = timer.end_session()
        
        assert session_data['completed'] == True
        assert session_data['mistakes'] == 1
        assert 'total_time' in session_data
        assert 'avg_reaction_time' in session_data
        assert len(timer.sessions) == 1
        assert timer.current_session is None
    
    def test_get_all_sessions(self):
        """Test getting all completed sessions."""
        timer = SessionTimer()
        
        # Start and end multiple sessions
        timer.start_session(3)
        timer.record_move()
        session1 = timer.end_session()
        
        timer.start_session(4)
        timer.record_move()
        session2 = timer.end_session()
        
        all_sessions = timer.get_all_sessions()
        assert len(all_sessions) == 2
        assert all_sessions[0]['grid_size'] == 3
        assert all_sessions[1]['grid_size'] == 4


class TestCard:
    """Test cases for the Card class."""
    
    def test_card_initialization(self):
        """Test card initialization."""
        card = Card(id=1, value="🐶", row=0, col=0)
        assert card.id == 1
        assert card.value == "🐶"
        assert card.state == CardState.HIDDEN
        assert card.row == 0
        assert card.col == 0
    
    def test_card_reveal(self):
        """Test revealing a card."""
        card = Card(id=1, value="🐶")
        assert card.is_hidden()
        
        card.reveal()
        assert card.is_revealed()
        assert not card.is_hidden()
    
    def test_card_hide(self):
        """Test hiding a card."""
        card = Card(id=1, value="🐶")
        card.reveal()
        assert card.is_revealed()
        
        card.hide()
        assert card.is_hidden()
        assert not card.is_revealed()
    
    def test_card_match(self):
        """Test matching a card."""
        card = Card(id=1, value="🐶")
        card.reveal()
        card.match()
        
        assert card.is_matched()
        assert not card.is_hidden()
        assert not card.is_revealed()


class TestMemoryGame:
    """Test cases for the MemoryGame class."""
    
    def test_game_initialization(self):
        """Test game initialization."""
        game = MemoryGame(4)
        assert game.grid_size == 4
        assert game.total_pairs == 8
        assert game.moves == 0
        assert game.mistakes == 0
        assert game.matched_pairs == 0
        assert game.game_state == GameState.WAITING
        assert len(game.cards) == 16
    
    def test_start_game(self):
        """Test starting a game."""
        game = MemoryGame(4)
        game.start_game()
        
        assert game.game_state == GameState.PLAYING
        assert game.moves == 0
        assert game.mistakes == 0
        assert game.matched_pairs == 0
        assert len(game.revealed_cards) == 0
    
    def test_get_card_at(self):
        """Test getting card at specific position."""
        game = MemoryGame(3)
        game.start_game()
        
        card = game.get_card_at(0, 0)
        assert card is not None
        assert card.row == 0
        assert card.col == 0
        
        # Test invalid position
        card = game.get_card_at(10, 10)
        assert card is None
    
    def test_click_card(self):
        """Test clicking a card."""
        game = MemoryGame(3)
        game.start_game()
        
        result = game.click_card(0, 0)
        
        assert "error" not in result
        assert result["card_revealed"] == True
        assert result["moves"] == 1
        assert len(game.revealed_cards) == 1
    
    def test_click_invalid_card(self):
        """Test clicking invalid cards."""
        game = MemoryGame(3)
        game.start_game()
        
        # Click already revealed card
        game.click_card(0, 0)
        result = game.click_card(0, 0)
        assert "error" in result
        
        # Click invalid position
        result = game.click_card(10, 10)
        assert "error" in result
    
    def test_card_matching(self):
        """Test card matching logic."""
        game = MemoryGame(2)  # 2x2 grid for easier testing
        game.start_game()
        
        # Find two cards with the same value
        card1 = game.get_card_at(0, 0)
        card2 = None
        
        for card in game.cards:
            if card.value == card1.value and card.id != card1.id:
                card2 = card
                break
        
        # Click first card
        game.click_card(card1.row, card1.col)
        
        # Click second card
        result = game.click_card(card2.row, card2.col)
        
        assert result.get("match_found") == True
        assert game.matched_pairs == 1
        assert len(game.revealed_cards) == 0
    
    def test_no_match(self):
        """Test when cards don't match."""
        game = MemoryGame(3)
        game.start_game()
        
        # Find two different cards
        card1 = game.get_card_at(0, 0)
        card2 = None
        
        for card in game.cards:
            if card.value != card1.value:
                card2 = card
                break
        
        # Click first card
        game.click_card(card1.row, card1.col)
        
        # Click second card
        result = game.click_card(card2.row, card2.col)
        
        assert result.get("match_found") == False
        assert game.mistakes == 1
        assert len(game.revealed_cards) == 2
    
    def test_hide_unmatched_cards(self):
        """Test hiding unmatched cards."""
        game = MemoryGame(3)
        game.start_game()
        
        # Reveal two different cards
        card1 = game.get_card_at(0, 0)
        card2 = None
        
        for card in game.cards:
            if card.value != card1.value:
                card2 = card
                break
        
        game.click_card(card1.row, card1.col)
        game.click_card(card2.row, card2.col)
        
        # Hide unmatched cards
        positions = game.hide_unmatched_cards()
        
        assert len(positions) == 2
        assert len(game.revealed_cards) == 0
    
    def test_game_completion(self):
        """Test game completion."""
        game = MemoryGame(2)  # 2x2 grid for easier testing
        game.start_game()
        
        # Complete the game by matching all pairs
        pairs = {}
        for card in game.cards:
            if card.value not in pairs:
                pairs[card.value] = []
            pairs[card.value].append(card)
        
        # Match all pairs
        for value, cards in pairs.items():
            game.click_card(cards[0].row, cards[0].col)
            game.click_card(cards[1].row, cards[1].col)
        
        assert game.is_game_complete() == True
        assert game.game_state == GameState.COMPLETED
    
    def test_get_game_stats(self):
        """Test getting game statistics."""
        game = MemoryGame(3)
        game.start_game()
        
        # Make some moves
        game.click_card(0, 0)
        game.click_card(0, 1)
        
        stats = game.get_game_stats()
        
        assert 'total_time' in stats
        assert 'moves' in stats
        assert 'mistakes' in stats
        assert 'matched_pairs' in stats
        assert 'total_pairs' in stats
        assert 'completion_percentage' in stats
        assert 'avg_reaction_time' in stats
    
    def test_get_board_state(self):
        """Test getting board state."""
        game = MemoryGame(3)
        game.start_game()
        
        board_state = game.get_board_state()
        
        assert len(board_state) == 3
        assert len(board_state[0]) == 3
        
        for row in board_state:
            for card_data in row:
                assert 'value' in card_data
                assert 'state' in card_data
                assert 'is_hidden' in card_data
                assert 'is_revealed' in card_data
                assert 'is_matched' in card_data


class TestPatternLearner:
    """Test cases for the PatternLearner class."""
    
    def test_learner_initialization(self):
        """Test pattern learner initialization."""
        learner = PatternLearner()
        assert learner.is_trained == False
        assert len(learner.feature_names) == 7
    
    def test_load_data_empty(self):
        """Test loading empty data."""
        learner = PatternLearner("tests/test_data.json")
        data = learner.load_data()
        
        assert "sessions" in data
        assert "user_profiles" in data
        assert "model_data" in data
        assert len(data["sessions"]) == 0
    
    def test_add_session(self):
        """Test adding a session."""
        learner = PatternLearner("tests/test_data.json")
        
        session_data = {
            "session_id": 1,
            "grid_size": 4,
            "total_time": 120.5,
            "avg_reaction_time": 1.2,
            "std_reaction_time": 0.5,
            "total_moves": 20,
            "mistakes": 3,
            "completion_percentage": 100.0,
            "completed": True
        }
        
        learner.add_session(session_data)
        data = learner.load_data()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["session_id"] == 1
    
    def test_extract_features(self):
        """Test feature extraction."""
        learner = PatternLearner()
        
        session_data = {
            "avg_reaction_time": 1.5,
            "std_reaction_time": 0.8,
            "total_moves": 25,
            "mistakes": 4,
            "total_time": 180.0,
            "grid_size": 4,
            "completion_percentage": 100.0
        }
        
        features = learner.extract_features(session_data)
        assert len(features) == 7
        assert features[0] == 1.5  # avg_reaction_time
        assert features[1] == 0.8  # std_reaction_time
        assert features[2] == 25   # total_moves
        assert features[3] == 4    # mistakes
        assert features[4] == 180.0  # total_time
        assert features[5] == 4    # grid_size
        assert features[6] == 100.0  # completion_percentage
    
    def test_calculate_performance_score(self):
        """Test performance score calculation."""
        learner = PatternLearner()
        
        # Good performance
        good_session = {
            "total_time": 60.0,
            "mistakes": 2,
            "avg_reaction_time": 1.0
        }
        good_score = learner._calculate_performance_score(good_session)
        assert good_score > 0.7
        
        # Poor performance
        poor_session = {
            "total_time": 400.0,
            "mistakes": 15,
            "avg_reaction_time": 5.0
        }
        poor_score = learner._calculate_performance_score(poor_session)
        assert poor_score < 0.3
    
    def test_determine_difficulty_level(self):
        """Test difficulty level determination."""
        learner = PatternLearner()
        
        assert learner._determine_difficulty_level(0.9) == DifficultyLevel.EASY
        assert learner._determine_difficulty_level(0.7) == DifficultyLevel.MEDIUM
        assert learner._determine_difficulty_level(0.5) == DifficultyLevel.HARD
        assert learner._determine_difficulty_level(0.2) == DifficultyLevel.EXPERT
    
    def test_train_models_insufficient_data(self):
        """Test training with insufficient data."""
        learner = PatternLearner("tests/test_data.json")
        result = learner.train_models()
        
        assert result["status"] == "insufficient_data"
        assert "Need at least 5 completed sessions" in result["message"]
    
    def test_predict_difficulty_untrained(self):
        """Test difficulty prediction with untrained model."""
        learner = PatternLearner()
        
        session_data = {
            "avg_reaction_time": 1.5,
            "std_reaction_time": 0.8,
            "total_moves": 25,
            "mistakes": 4,
            "total_time": 180.0,
            "grid_size": 4,
            "completion_percentage": 100.0
        }
        
        result = learner.predict_difficulty(session_data)
        assert "error" in result
        assert result["recommended_difficulty"] == DifficultyLevel.MEDIUM


class TestAdaptiveDifficulty:
    """Test cases for the AdaptiveDifficulty class."""
    
    def test_adaptive_difficulty_initialization(self):
        """Test adaptive difficulty initialization."""
        learner = PatternLearner()
        adaptive = AdaptiveDifficulty(learner)
        
        assert adaptive.pattern_learner == learner
        assert len(adaptive.difficulty_settings) == 4
    
    def test_get_difficulty_settings(self):
        """Test getting difficulty settings."""
        learner = PatternLearner()
        adaptive = AdaptiveDifficulty(learner)
        
        easy_settings = adaptive.get_difficulty_settings(DifficultyLevel.EASY)
        assert easy_settings["grid_size"] == 3
        assert easy_settings["time_limit"] == 300
        
        medium_settings = adaptive.get_difficulty_settings(DifficultyLevel.MEDIUM)
        assert medium_settings["grid_size"] == 4
        assert medium_settings["time_limit"] == 240
    
    def test_evaluate_current_performance(self):
        """Test performance evaluation."""
        learner = PatternLearner()
        adaptive = AdaptiveDifficulty(learner)
        
        session_data = {
            "total_time": 120.0,
            "mistakes": 3,
            "avg_reaction_time": 1.5,
            "completion_percentage": 100.0
        }
        
        performance = adaptive._evaluate_current_performance(session_data)
        
        assert "overall_score" in performance
        assert "time_score" in performance
        assert "mistake_score" in performance
        assert "reaction_score" in performance
        assert "completion_percentage" in performance
        assert "performance_level" in performance
        assert 0 <= performance["overall_score"] <= 1
    
    def test_get_performance_level(self):
        """Test performance level determination."""
        learner = PatternLearner()
        adaptive = AdaptiveDifficulty(learner)
        
        assert adaptive._get_performance_level(0.9) == "excellent"
        assert adaptive._get_performance_level(0.7) == "good"
        assert adaptive._get_performance_level(0.5) == "average"
        assert adaptive._get_performance_level(0.2) == "needs_improvement"
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        learner = PatternLearner()
        adaptive = AdaptiveDifficulty(learner)
        
        # Good performance
        good_session = {
            "avg_reaction_time": 1.0,
            "mistakes": 2,
            "total_time": 90.0
        }
        good_recs = adaptive._generate_recommendations(good_session)
        assert len(good_recs) > 0
        
        # Poor performance
        poor_session = {
            "avg_reaction_time": 4.0,
            "mistakes": 10,
            "total_time": 400.0
        }
        poor_recs = adaptive._generate_recommendations(poor_session)
        assert len(poor_recs) > 0
    
    def test_rule_based_difficulty_adjustment(self):
        """Test rule-based difficulty adjustment."""
        learner = PatternLearner()
        adaptive = AdaptiveDifficulty(learner)
        
        # Good performance
        good_session = {
            "total_time": 60.0,
            "mistakes": 2,
            "avg_reaction_time": 1.0,
            "completion_percentage": 100.0
        }
        good_result = adaptive._rule_based_difficulty_adjustment(good_session)
        assert good_result["recommended_difficulty"] in [DifficultyLevel.HARD, DifficultyLevel.MEDIUM]
        
        # Poor performance
        poor_session = {
            "total_time": 400.0,
            "mistakes": 15,
            "avg_reaction_time": 5.0,
            "completion_percentage": 50.0
        }
        poor_result = adaptive._rule_based_difficulty_adjustment(poor_session)
        assert poor_result["recommended_difficulty"] == DifficultyLevel.EASY
    
    def test_get_adaptive_hints(self):
        """Test adaptive hints generation."""
        learner = PatternLearner()
        adaptive = AdaptiveDifficulty(learner)
        
        session_data = {
            "avg_reaction_time": 4.0,
            "mistakes": 8,
            "total_moves": 30,
            "grid_size": 4
        }
        
        hints = adaptive.get_adaptive_hints(session_data)
        assert len(hints) > 0
        assert any("reaction time" in hint.lower() for hint in hints)
    
    def test_get_personalized_feedback(self):
        """Test personalized feedback generation."""
        learner = PatternLearner()
        adaptive = AdaptiveDifficulty(learner)
        
        session_data = {
            "total_time": 120.0,
            "mistakes": 3,
            "avg_reaction_time": 1.5,
            "completion_percentage": 100.0
        }
        
        feedback = adaptive.get_personalized_feedback(session_data)
        
        assert "performance_summary" in feedback
        assert "strengths" in feedback
        assert "areas_for_improvement" in feedback
        assert "next_session_recommendation" in feedback
        assert "motivational_message" in feedback


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_complete_game_flow(self):
        """Test a complete game flow."""
        # Initialize components
        game = MemoryGame(3)
        timer = SessionTimer()
        learner = PatternLearner("tests/test_data.json")
        adaptive = AdaptiveDifficulty(learner)
        
        # Start game
        game.start_game()
        timer.start_session(3)
        
        # Make some moves
        game.click_card(0, 0)
        timer.record_move()
        
        game.click_card(0, 1)
        timer.record_move(is_mistake=True)
        
        # Complete game (simplified)
        game.matched_pairs = game.total_pairs
        game.game_state = GameState.COMPLETED
        
        # End session
        session_data = timer.end_session()
        final_stats = game.get_game_stats()
        complete_session = {**session_data, **final_stats, "completed": True}
        
        # Add to learner
        learner.add_session(complete_session)
        
        # Test adaptive difficulty
        feedback = adaptive.get_personalized_feedback(complete_session)
        assert "performance_summary" in feedback
    
    def test_ml_training_flow(self):
        """Test ML training flow with multiple sessions."""
        learner = PatternLearner("tests/test_data.json")
        
        # Create multiple training sessions
        for i in range(6):
            session_data = {
                "session_id": i + 1,
                "grid_size": 4,
                "total_time": 120.0 + i * 10,
                "avg_reaction_time": 1.5 + i * 0.1,
                "std_reaction_time": 0.5,
                "total_moves": 20 + i,
                "mistakes": 3 + (i % 3),
                "completion_percentage": 100.0,
                "completed": True
            }
            learner.add_session(session_data)
        
        # Train model
        result = learner.train_models()
        assert result["status"] == "success"
        assert "difficulty_accuracy" in result
        assert "performance_mse" in result
    
    def test_adaptive_difficulty_with_ml(self):
        """Test adaptive difficulty with trained ML model."""
        learner = PatternLearner("tests/test_data.json")
        adaptive = AdaptiveDifficulty(learner)
        
        # Train model first
        for i in range(6):
            session_data = {
                "session_id": i + 1,
                "grid_size": 4,
                "total_time": 120.0 + i * 10,
                "avg_reaction_time": 1.5 + i * 0.1,
                "std_reaction_time": 0.5,
                "total_moves": 20 + i,
                "mistakes": 3 + (i % 3),
                "completion_percentage": 100.0,
                "completed": True
            }
            learner.add_session(session_data)
        
        learner.train_models()
        
        # Test difficulty prediction
        test_session = {
            "avg_reaction_time": 1.5,
            "std_reaction_time": 0.8,
            "total_moves": 25,
            "mistakes": 4,
            "total_time": 180.0,
            "grid_size": 4,
            "completion_percentage": 100.0
        }
        
        prediction = adaptive.suggest_next_difficulty(test_session)
        assert "recommended_difficulty" in prediction
        assert "confidence" in prediction
        assert "settings" in prediction


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 