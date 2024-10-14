import random

def roll_die():
    return random.randint(1, 6)

def player_turn(player_name, total_score):
    turn_score = 0
    while True:
        roll = roll_die()
        print(f"{player_name} rolled a {roll}")
        if roll == 1:
            print(f"{player_name} rolled a 1! Turn over. No points added.")
            return total_score
        else:
            turn_score += roll
            print(f"{player_name}'s turn score is now {turn_score}")
            choice = input("Do you want to 'roll' again or 'hold'? ").lower()
            if choice == 'hold':
                total_score += turn_score
                print(f"{player_name} holds. Total score is now {total_score}")
                return total_score

def pig_game():
    player_1 = input("Enter Player 1's name: ")
    player_2 = input("Enter Player 2's name: ")

    score_1 = 0
    score_2 = 0
    winning_score = 100

    while score_1 < winning_score and score_2 < winning_score:
        print(f"\n{player_1}'s turn:")
        score_1 = player_turn(player_1, score_1)
        if score_1 >= winning_score:
            print(f"\n{player_1} wins with a score of {score_1}!")
            break

        print(f"\n{player_2}'s turn:")
        score_2 = player_turn(player_2, score_2)
        if score_2 >= winning_score:
            print(f"\n{player_2} wins with a score of {score_2}!")
            break

if __name__ == "__main__":
    pig_game()
