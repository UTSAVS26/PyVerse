import json
import os
from typing import Dict, List, Tuple, Optional
import streamlit as st
import pandas as pd

class PromptInterface:
    """Handles the interaction with users through prompts and collects responses."""
    
    def __init__(self, questions_file: str = "prompts/questions.json"):
        """Initialize the prompt interface with questions."""
        self.questions_file = questions_file
        self.questions = self._load_questions()
    
    def _load_questions(self) -> List[Dict]:
        """Load questions from JSON file."""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('questions', [])
        except FileNotFoundError:
            print(f"Error: Questions file {self.questions_file} not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.questions_file}")
            return []
    
    def get_questions(self) -> List[Dict]:
        """Get all available questions."""
        return self.questions
    
    def run_cli_interface(self) -> Dict[int, str]:
        """Run the command-line interface to collect user responses."""
        responses = {}
        
        print("🧠 Welcome to TextPersona - Personality Type Predictor!")
        print("=" * 50)
        print("Please answer the following questions thoughtfully.\n")
        
        for question in self.questions:
            print(f"\nQuestion {question['id']}: {question['question']}")
            print("Options:")
            for i, option in enumerate(question['options'], 1):
                print(f"  {i}. {option}")
            
            while True:
                try:
                    choice = input("\nYour choice (1-3): ").strip()
                    if choice in ['1', '2', '3']:
                        selected_option = question['options'][int(choice) - 1]
                        responses[question['id']] = selected_option
                        break
                    else:
                        print("Please enter 1, 2, or 3.")
                except (ValueError, IndexError):
                    print("Please enter a valid number (1-3).")
        
        return responses
    
    def run_streamlit_interface(self) -> Optional[Dict[int, str]]:
        """Run the Streamlit interface to collect user responses."""
        st.title("🧠 TextPersona - Personality Type Predictor")
        st.markdown("---")
        
        responses = {}
        
        # Create a form for all questions
        with st.form("personality_quiz"):
            st.subheader("Please answer the following questions thoughtfully:")
            
            for question in self.questions:
                st.write(f"**Question {question['id']}:** {question['question']}")
                
                # Create radio buttons for options
                option_labels = [f"{i+1}. {option}" for i, option in enumerate(question['options'])]
                choice = st.radio(
                    "Select your answer:",
                    option_labels,
                    key=f"q{question['id']}",
                    label_visibility="collapsed"
                )
                
                # Extract the selected option
                if choice:
                    option_index = int(choice.split('.')[0]) - 1
                    selected_option = question['options'][option_index]
                    responses[question['id']] = selected_option
            
            submitted = st.form_submit_button("Submit Answers")
            
            if submitted:
                if len(responses) == len(self.questions):
                    return responses
                else:
                    st.error("Please answer all questions before submitting.")
                    return None
        
        return None
    
    def format_responses_for_classifier(self, responses: Dict[int, str]) -> str:
        """Format responses into a single text for classification."""
        formatted_text = "Personality Assessment Responses:\n\n"
        
        for question in self.questions:
            q_id = question['id']
            if q_id in responses:
                formatted_text += f"Q{q_id}: {question['question']}\n"
                formatted_text += f"A{q_id}: {responses[q_id]}\n\n"
        
        return formatted_text
    
    def save_responses(self, responses: Dict[int, str], filename: str = "data/user_logs.json"):
        """Save user responses to a JSON file (anonymously)."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Load existing logs or create new
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = {"responses": []}
            
            # Add new response (without personal identifiers)
            logs["responses"].append({
                "timestamp": str(pd.Timestamp.now()),
                "responses": responses
            })
            
            # Save updated logs
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save responses: {e}")

# For CLI usage
if __name__ == "__main__":
    import pandas as pd
    interface = PromptInterface()
    responses = interface.run_cli_interface()
    
    if responses:
        print("\n" + "=" * 50)
        print("Your Responses:")
        for q_id, response in responses.items():
            print(f"Q{q_id}: {response}")
        
        # Save responses
        interface.save_responses(responses) 