"""
Pattern detection and analysis for habit tracking data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from src.models.habit_model import HabitTracker, HabitEntry


class PatternDetector:
    """Analyzes habit patterns and provides insights using machine learning."""
    
    def __init__(self, tracker: HabitTracker):
        """Initialize pattern detector with habit tracker data."""
        self.tracker = tracker
        self.df = tracker.to_dataframe()
        self.correlation_matrix = None
        self.feature_importance = None
        self.predictions = {}
        
    def analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between habits and outcomes."""
        if self.df.empty:
            return {}
        
        # Calculate correlation matrix
        numeric_cols = ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                       'water_glasses', 'work_hours', 'mood_rating', 'productivity_rating']
        
        correlation_matrix = self.df[numeric_cols].corr()
        self.correlation_matrix = correlation_matrix
        
        # Extract key correlations with mood and productivity
        mood_correlations = correlation_matrix['mood_rating'].drop('mood_rating')
        productivity_correlations = correlation_matrix['productivity_rating'].drop('productivity_rating')
        
        return {
            'mood_correlations': mood_correlations.to_dict(),
            'productivity_correlations': productivity_correlations.to_dict(),
            'correlation_matrix': correlation_matrix.to_dict()
        }
    
    def detect_patterns(self) -> Dict[str, Any]:
        """Detect patterns in the habit data."""
        if self.df.empty or len(self.df) < 7:  # Need at least a week of data
            return {'message': 'Need at least 7 days of data for pattern detection'}
        
        patterns = {}
        
        # Sleep patterns
        patterns['sleep'] = self._analyze_sleep_patterns()
        
        # Exercise patterns
        patterns['exercise'] = self._analyze_exercise_patterns()
        
        # Screen time patterns
        patterns['screen_time'] = self._analyze_screen_time_patterns()
        
        # Weekly patterns
        patterns['weekly'] = self._analyze_weekly_patterns()
        
        # Streak analysis
        patterns['streaks'] = self._analyze_streaks()
        
        return patterns
    
    def _analyze_sleep_patterns(self) -> Dict[str, Any]:
        """Analyze sleep patterns and their impact."""
        sleep_data = self.df[['sleep_hours', 'mood_rating', 'productivity_rating']].dropna()
        
        if len(sleep_data) < 3:
            return {}
        
        # Optimal sleep range
        optimal_sleep = sleep_data.groupby(pd.cut(sleep_data['sleep_hours'], 
                                                 bins=[0, 6, 7, 8, 9, 24])).agg({
            'mood_rating': 'mean',
            'productivity_rating': 'mean'
        }).round(2)
        
        # Find best sleep range
        best_mood_sleep = optimal_sleep['mood_rating'].idxmax()
        best_productivity_sleep = optimal_sleep['productivity_rating'].idxmax()
        
        return {
            'optimal_sleep_ranges': optimal_sleep.to_dict(),
            'best_mood_sleep_range': str(best_mood_sleep),
            'best_productivity_sleep_range': str(best_productivity_sleep),
            'avg_sleep': sleep_data['sleep_hours'].mean(),
            'sleep_consistency': sleep_data['sleep_hours'].std()
        }
    
    def _analyze_exercise_patterns(self) -> Dict[str, Any]:
        """Analyze exercise patterns and their impact."""
        exercise_data = self.df[['exercise_minutes', 'mood_rating', 'productivity_rating']].dropna()
        
        if len(exercise_data) < 3:
            return {}
        
        # Exercise vs no exercise
        exercise_days = exercise_data[exercise_data['exercise_minutes'] > 0]
        no_exercise_days = exercise_data[exercise_data['exercise_minutes'] == 0]
        
        exercise_impact = {}
        if len(exercise_days) > 0 and len(no_exercise_days) > 0:
            exercise_impact = {
                'exercise_days_mood': exercise_days['mood_rating'].mean(),
                'no_exercise_days_mood': no_exercise_days['mood_rating'].mean(),
                'exercise_days_productivity': exercise_days['productivity_rating'].mean(),
                'no_exercise_days_productivity': no_exercise_days['productivity_rating'].mean(),
                'mood_improvement': exercise_days['mood_rating'].mean() - no_exercise_days['mood_rating'].mean(),
                'productivity_improvement': exercise_days['productivity_rating'].mean() - no_exercise_days['productivity_rating'].mean()
            }
        
        return {
            'exercise_impact': exercise_impact,
            'avg_exercise_minutes': exercise_data['exercise_minutes'].mean(),
            'exercise_frequency': len(exercise_days) / len(exercise_data),
            'exercise_consistency': exercise_data['exercise_minutes'].std()
        }
    
    def _analyze_screen_time_patterns(self) -> Dict[str, Any]:
        """Analyze screen time patterns and their impact."""
        screen_data = self.df[['screen_time_hours', 'mood_rating', 'productivity_rating']].dropna()
        
        if len(screen_data) < 3:
            return {}
        
        # High vs low screen time
        high_screen = screen_data[screen_data['screen_time_hours'] > 6]
        low_screen = screen_data[screen_data['screen_time_hours'] <= 6]
        
        screen_impact = {}
        if len(high_screen) > 0 and len(low_screen) > 0:
            screen_impact = {
                'high_screen_mood': high_screen['mood_rating'].mean(),
                'low_screen_mood': low_screen['mood_rating'].mean(),
                'high_screen_productivity': high_screen['productivity_rating'].mean(),
                'low_screen_productivity': low_screen['productivity_rating'].mean(),
                'mood_difference': low_screen['mood_rating'].mean() - high_screen['mood_rating'].mean(),
                'productivity_difference': low_screen['productivity_rating'].mean() - high_screen['productivity_rating'].mean()
            }
        
        return {
            'screen_impact': screen_impact,
            'avg_screen_time': screen_data['screen_time_hours'].mean(),
            'screen_time_consistency': screen_data['screen_time_hours'].std()
        }
    
    def _analyze_weekly_patterns(self) -> Dict[str, Any]:
        """Analyze weekly patterns in the data."""
        if len(self.df) < 7:
            return {}
        
        # Add day of week
        df_with_dow = self.df.copy()
        df_with_dow['day_of_week'] = df_with_dow['date'].dt.day_name()
        
        # Group by day of week
        weekly_patterns = df_with_dow.groupby('day_of_week').agg({
            'mood_rating': 'mean',
            'productivity_rating': 'mean',
            'sleep_hours': 'mean',
            'exercise_minutes': 'mean',
            'screen_time_hours': 'mean'
        }).round(2)
        
        return {
            'weekly_averages': weekly_patterns.to_dict(),
            'best_mood_day': weekly_patterns['mood_rating'].idxmax(),
            'best_productivity_day': weekly_patterns['productivity_rating'].idxmax(),
            'worst_mood_day': weekly_patterns['mood_rating'].idxmin(),
            'worst_productivity_day': weekly_patterns['productivity_rating'].idxmin()
        }
    
    def _analyze_streaks(self) -> Dict[str, Any]:
        """Analyze streaks of good/bad habits."""
        if len(self.df) < 3:
            return {}
        
        # Exercise streaks
        exercise_streaks = self._calculate_streaks(self.df['exercise_minutes'] > 0)
        
        # Good sleep streaks (7+ hours)
        good_sleep_streaks = self._calculate_streaks(self.df['sleep_hours'] >= 7)
        
        # High productivity streaks (4+ rating)
        high_productivity_streaks = self._calculate_streaks(self.df['productivity_rating'] >= 4)
        
        return {
            'exercise_streaks': exercise_streaks,
            'good_sleep_streaks': good_sleep_streaks,
            'high_productivity_streaks': high_productivity_streaks
        }
    
    def _calculate_streaks(self, condition: pd.Series) -> Dict[str, Any]:
        """Calculate streaks for a given condition."""
        streaks = []
        current_streak = 0
        
        for value in condition:
            if value:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        if not streaks:
            return {'max_streak': 0, 'avg_streak': 0, 'total_streaks': 0}
        
        return {
            'max_streak': max(streaks),
            'avg_streak': sum(streaks) / len(streaks),
            'total_streaks': len(streaks)
        }
    
    def predict_productivity(self, days_ahead: int = 7) -> Dict[str, Any]:
        """Predict productivity for the next few days."""
        if len(self.df) < 14:  # Need at least 2 weeks of data
            return {'message': 'Need at least 14 days of data for predictions'}
        
        # Prepare features
        features = ['sleep_hours', 'exercise_minutes', 'screen_time_hours', 
                   'water_glasses', 'work_hours']
        
        X = self.df[features].dropna()
        y = self.df['productivity_rating'].dropna()
        
        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) < 10:
            return {'message': 'Insufficient data for prediction'}
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Feature importance
        self.feature_importance = dict(zip(features, model.feature_importances_))
        
        # Make predictions based on recent averages
        recent_avg = X.tail(7).mean()
        predicted_productivity = model.predict([recent_avg])[0]
        
        # Confidence based on model performance
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        return {
            'predicted_productivity': round(predicted_productivity, 2),
            'confidence': round(r2, 2),
            'feature_importance': self.feature_importance,
            'recent_averages': recent_avg.to_dict()
        }
    
    def generate_insights(self) -> List[str]:
        """Generate actionable insights from the analysis."""
        insights = []
        
        if self.df.empty:
            return ["Start logging your habits to get personalized insights!"]
        
        # Get correlations
        correlations = self.analyze_correlations()
        mood_corr = correlations.get('mood_correlations', {})
        productivity_corr = correlations.get('productivity_correlations', {})
        
        # Sleep insights
        if 'sleep_hours' in mood_corr and abs(mood_corr['sleep_hours']) > 0.3:
            if mood_corr['sleep_hours'] > 0:
                insights.append("More sleep correlates with better mood!")
            else:
                insights.append("Less sleep correlates with better mood - consider your optimal sleep pattern.")
        
        # Exercise insights
        if 'exercise_minutes' in mood_corr and mood_corr['exercise_minutes'] > 0.2:
            insights.append("Exercise days show significantly better mood!")
        
        # Screen time insights
        if 'screen_time_hours' in mood_corr and mood_corr['screen_time_hours'] < -0.2:
            insights.append("High screen time correlates with lower mood - consider digital detox.")
        
        # Get patterns
        patterns = self.detect_patterns()
        
        # Sleep patterns
        sleep_patterns = patterns.get('sleep', {})
        if sleep_patterns:
            avg_sleep = sleep_patterns.get('avg_sleep', 0)
            if avg_sleep < 7:
                insights.append(f"Your average sleep is {avg_sleep:.1f} hours - consider aiming for 7-9 hours.")
            elif avg_sleep > 9:
                insights.append(f"Your average sleep is {avg_sleep:.1f} hours - this might be too much.")
        
        # Exercise patterns
        exercise_patterns = patterns.get('exercise', {})
        if exercise_patterns:
            exercise_impact = exercise_patterns.get('exercise_impact', {})
            if exercise_impact and exercise_impact.get('mood_improvement', 0) > 0.5:
                insights.append("Exercise days show much better mood - try to exercise regularly!")
        
        # Weekly patterns
        weekly_patterns = patterns.get('weekly', {})
        if weekly_patterns:
            best_day = weekly_patterns.get('best_productivity_day')
            worst_day = weekly_patterns.get('worst_productivity_day')
            if best_day and worst_day:
                insights.append(f"Your most productive day is {best_day}, least productive is {worst_day}.")
        
        # Streaks
        streaks = patterns.get('streaks', {})
        exercise_streaks = streaks.get('exercise_streaks', {})
        if exercise_streaks and exercise_streaks.get('max_streak', 0) > 0:
            insights.append(f"Your longest exercise streak is {exercise_streaks['max_streak']} days!")
        
        if not insights:
            insights.append("Keep logging your habits to discover more patterns!")
        
        return insights
    
    def get_recommendations(self) -> List[str]:
        """Get personalized recommendations based on patterns."""
        recommendations = []
        
        if self.df.empty:
            return ["Start by logging your daily habits for at least a week to get personalized recommendations."]
        
        # Get patterns and correlations
        patterns = self.detect_patterns()
        correlations = self.analyze_correlations()
        mood_corr = correlations.get('mood_correlations', {})
        productivity_corr = correlations.get('productivity_correlations', {})
        
        # Sleep recommendations
        sleep_patterns = patterns.get('sleep', {})
        if sleep_patterns:
            best_sleep_range = sleep_patterns.get('best_mood_sleep_range', '')
            if best_sleep_range:
                recommendations.append(f"Your optimal sleep range appears to be {best_sleep_range} hours for best mood.")
        
        # Exercise recommendations
        exercise_patterns = patterns.get('exercise', {})
        if exercise_patterns:
            exercise_impact = exercise_patterns.get('exercise_impact', {})
            if exercise_impact and exercise_impact.get('mood_improvement', 0) > 0.3:
                recommendations.append("Exercise significantly improves your mood. Try to exercise at least 3 times per week.")
        
        # Screen time recommendations
        screen_patterns = patterns.get('screen_time', {})
        if screen_patterns:
            screen_impact = screen_patterns.get('screen_impact', {})
            if screen_impact and screen_impact.get('mood_difference', 0) > 0.5:
                recommendations.append("Limiting screen time to under 6 hours per day could improve your mood significantly.")
        
        # Productivity recommendations
        if 'sleep_hours' in productivity_corr and productivity_corr['sleep_hours'] > 0.3:
            recommendations.append("Getting adequate sleep is crucial for your productivity. Prioritize sleep!")
        
        if 'exercise_minutes' in productivity_corr and productivity_corr['exercise_minutes'] > 0.2:
            recommendations.append("Regular exercise boosts your productivity. Consider morning workouts!")
        
        # Weekly planning recommendations
        weekly_patterns = patterns.get('weekly', {})
        if weekly_patterns:
            best_day = weekly_patterns.get('best_productivity_day')
            if best_day:
                recommendations.append(f"Schedule your most important tasks for {best_day}s when you're most productive.")
        
        if not recommendations:
            recommendations.append("Continue tracking your habits to receive more personalized recommendations.")
        
        return recommendations
