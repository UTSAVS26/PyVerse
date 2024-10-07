import streamlit as st
import random

def number_guessing_game():
    st.title("Number Guessing Game")

    # Initialize session state to store target number and attempts
    if "target_number" not in st.session_state:
        st.session_state.target_number = random.randint(1, 100)
        st.session_state.attempts = 0
        st.session_state.game_over = False

    # Game instructions
    st.write("Guess the number I'm thinking of! It's between 1 and 100.")

    # Input form for user guesses
    with st.form("guess_form"):
        guess = st.number_input("Enter your guess:", min_value=1, max_value=100, step=1)
        submitted = st.form_submit_button("Submit Guess")

    if submitted:
        if st.session_state.game_over:
            st.warning("The game is over! Click the 'Restart' button to play again.")
        else:
            st.session_state.attempts += 1
            if guess < st.session_state.target_number:
                st.warning("Too low! Try again.")
            elif guess > st.session_state.target_number:
                st.warning("Too high! Try again.")
            else:
                st.success(f"Congratulations! You guessed the number in {st.session_state.attempts} attempts.")
                st.session_state.game_over = True

    # Restart button to start a new game
    if st.button("Restart"):
        st.session_state.target_number = random.randint(1, 100)
        st.session_state.attempts = 0
        st.session_state.game_over = False
        st.info("New game started! Guess the new number.")

# Run the game
if __name__ == "__main__":
    number_guessing_game()
