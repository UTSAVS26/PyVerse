import random
import streamlit as st

class Player:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.treasure = 0

    def attack(self):
        return random.randint(10, 20)

    def take_damage(self, damage):
        self.health -= damage
        if self.health < 0:
            self.health = 0

class Monster:
    def __init__(self):
        self.health = random.randint(30, 60)

    def attack(self):
        return random.randint(5, 15)

def encounter_monster(player):
    # Manage a single monster encounter step by step.
    if 'monster' not in st.session_state:
        st.session_state.monster = Monster()
        st.session_state.encounter_log = ["A wild monster appears!"]

    monster = st.session_state.monster
    encounter_log = []

    # Player's turn
    if st.session_state.turn == 'player':
        damage = player.attack()
        monster.health -= damage
        encounter_log.append(f"You attack the monster for {damage} damage.")
        encounter_log.append(f"Monster's health: {monster.health}")

        if monster.health <= 0:
            encounter_log.append("You defeated the monster!")
            treasure_found = random.randint(10, 50)
            player.treasure += treasure_found
            encounter_log.append(f"You found {treasure_found} treasure!")
            del st.session_state.monster  # Monster defeated, remove from session state
            st.session_state.turn = 'explore'
        else:
            st.session_state.turn = 'monster'

    # Monster's turn
    elif st.session_state.turn == 'monster':
        damage = monster.attack()
        player.take_damage(damage)
        encounter_log.append(f"The monster attacks you for {damage} damage.")
        encounter_log.append(f"Your health: {player.health}")

        if player.health <= 0:
            encounter_log.append("You have been defeated. Game over!")
            st.session_state.game_over = True
        else:
            st.session_state.turn = 'player'

    return encounter_log

def explore_dungeon(player):
    if 'turn' not in st.session_state:
        st.session_state.turn = 'explore'

    encounter_log = []

    if st.session_state.turn == 'explore':
        encounter_log.append("You explore the dungeon...")

        # Random encounter
        encounter = random.choice(["monster", "treasure", "nothing"])
        if encounter == "monster":
            st.session_state.turn = 'player'
            encounter_log += encounter_monster(player)
        elif encounter == "treasure":
            treasure_found = random.randint(5, 30)
            player.treasure += treasure_found
            encounter_log.append(f"You found {treasure_found} treasure!")
        else:
            encounter_log.append("You found nothing but dust...")
        
        return encounter_log
    else:
        return encounter_monster(player)

def main():
    st.title("Dungeon Exploration Game")

    # Input for player name
    if 'player' not in st.session_state:
        name = st.text_input("Enter your character's name:", "")
        if name:
            st.session_state['player'] = Player(name)
            st.session_state.turn = 'explore'
            st.session_state.game_over = False
            st.session_state.encounter_log = []

    player = st.session_state.get('player', None)

    if player and not st.session_state.game_over:
        st.write(f"Welcome, {player.name}!")
        st.write(f"Health: {player.health}")
        st.write(f"Treasure: {player.treasure}")

        if st.button("Next Step"):
            # Clear the log after each step to simulate screen clearing
            st.session_state.encounter_log = []
            encounter_log = explore_dungeon(player)
            st.session_state.encounter_log += encounter_log

        # Show only the latest log (clear previous content)
        for event in st.session_state.encounter_log:
            st.write(event)

        if player.health <= 0:
            st.write(f"Game Over! You collected {player.treasure} treasure.")
            if st.button("Restart Game"):
                del st.session_state['player']

if __name__ == "__main__":
    # Run the Streamlit app using the command:
    # streamlit run dungeon_game.py
    main()
