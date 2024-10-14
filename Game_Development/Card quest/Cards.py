import random

# Card Class
class Card:
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def value(self):
        if self.rank in ['Jack', 'Queen', 'King']:
            return 10
        elif self.rank == 'Ace':
            return 11
        else:
            return int(self.rank)

    def __str__(self):
        return f"{self.rank} of {self.suit}"

# Deck Class
class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in Card.suits for rank in Card.ranks]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop() if self.cards else None

# Hand Class
class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def total_value(self):
        value = sum(card.value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 'Ace')
        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value

    def __str__(self):
        return ', '.join(str(card) for card in self.cards)

# Blackjack Game Class
class BlackjackGame:
    def __init__(self):
        self.deck = Deck()
        self.player_hand = Hand()
        self.dealer_hand = Hand()

    def start_game(self):
        self.player_hand.add_card(self.deck.deal())
        self.player_hand.add_card(self.deck.deal())
        self.dealer_hand.add_card(self.deck.deal())
        self.dealer_hand.add_card(self.deck.deal())

        print("Dealer's hand: ", self.dealer_hand.cards[0])
        print("Your hand: ", self.player_hand)

        while True:
            action = input("Do you want to hit or stand? (h/s): ").lower()
            if action == 'h':
                self.player_hand.add_card(self.deck.deal())
                print("Your hand: ", self.player_hand)
                if self.player_hand.total_value() > 21:
                    print("You bust! Dealer wins.")
                    return
            elif action == 's':
                break
            else:
                print("Invalid input, please enter 'h' or 's'.")

        print("Dealer's hand: ", self.dealer_hand)
        while self.dealer_hand.total_value() < 17:
            self.dealer_hand.add_card(self.deck.deal())
            print("Dealer draws a card: ", self.dealer_hand)

        self.determine_winner()

    def determine_winner(self):
        player_total = self.player_hand.total_value()
        dealer_total = self.dealer_hand.total_value()
        print(f"Your total: {player_total}, Dealer's total: {dealer_total}")
        if dealer_total > 21 or player_total > dealer_total:
            print("You win!")
        elif player_total < dealer_total:
            print("Dealer wins.")
        else:
            print("It's a tie!")

# War Game Class
class WarGame:
    def __init__(self):
        self.deck = Deck()
        self.player_hand = Hand()
        self.dealer_hand = Hand()

    def start_game(self):
        while len(self.deck.cards) > 0:
            self.player_hand.add_card(self.deck.deal())
            self.dealer_hand.add_card(self.deck.deal())

        print("Player's hand size:", len(self.player_hand.cards))
        print("Dealer's hand size:", len(self.dealer_hand.cards))

        while self.player_hand.cards and self.dealer_hand.cards:
            player_card = self.player_hand.cards.pop(0)
            dealer_card = self.dealer_hand.cards.pop(0)
            print(f"Player plays: {player_card}, Dealer plays: {dealer_card}")

            if player_card.value() > dealer_card.value():
                print("Player wins this round!")
                self.player_hand.add_card(player_card)
                self.player_hand.add_card(dealer_card)
            elif player_card.value() < dealer_card.value():
                print("Dealer wins this round!")
                self.dealer_hand.add_card(player_card)
                self.dealer_hand.add_card(dealer_card)
            else:
                print("It's a tie! Both cards go to the bottom of their respective piles.")

        if self.player_hand.cards:
            print("Player wins the game!")
        else:
            print("Dealer wins the game!")

# Higher or Lower Game Class
class HigherLowerGame:
    def __init__(self):
        self.deck = Deck()

    def start_game(self):
        current_card = self.deck.deal()
        print(f"Current card: {current_card}")

        while True:
            next_card = self.deck.deal()
            print(f"Next card is {next_card}. Do you think it's higher or lower than the current card? (h/l): ", end="")
            guess = input().lower()

            if (guess == 'h' and next_card.value() > current_card.value()) or \
               (guess == 'l' and next_card.value() < current_card.value()):
                print("Correct! Next card is now the current card.")
                current_card = next_card
            else:
                print("Wrong! Game over.")
                break

# Memory Game Class
class MemoryGame:
    def __init__(self):
        self.deck = Deck()
        self.cards = self.deck.cards * 2  # Create pairs
        self.shuffle_cards()

    def shuffle_cards(self):
        random.shuffle(self.cards)

    def start_game(self):
        print("Welcome to Memory Game!")
        pairs = len(self.cards) // 2
        matched_pairs = 0
        revealed = []

        while matched_pairs < pairs:
            print(f"Revealed cards: {revealed}")
            print(f"You have {len(self.cards)} cards remaining.")
            print("Select two card indices to reveal (e.g., 0 1):")
            indices = list(map(int, input().split()))

            if len(indices) != 2 or any(i >= len(self.cards) for i in indices):
                print("Invalid indices. Try again.")
                continue

            revealed_cards = [self.cards[i] for i in indices]
            print(f"You revealed: {revealed_cards[0]} and {revealed_cards[1]}")

            if revealed_cards[0].value() == revealed_cards[1].value():
                print("It's a match!")
                matched_pairs += 1
                for i in indices:
                    revealed.append(self.cards[i])
                self.cards = [c for c in self.cards if c not in revealed]
            else:
                print("Not a match. Try again.")

        print("Congratulations! You've matched all pairs!")

# Go Fish Game Class
class GoFishGame:
    def __init__(self):
        self.deck = Deck()
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.pairs_found = 0

    def start_game(self):
        for _ in range(5):  # Deal 5 cards to each player
            self.player_hand.add_card(self.deck.deal())
            self.dealer_hand.add_card(self.deck.deal())

        print("Your hand: ", self.player_hand)
        
        while True:
            print("Ask for a rank (e.g., '2', 'Jack', etc.):")
            request = input().capitalize()

            if not any(card.rank == request for card in self.player_hand.cards):
                print("You don't have that rank! Go Fish.")
                self.player_hand.add_card(self.deck.deal())
                continue

            print(f"You asked for: {request}")
            dealer_matches = [card for card in self.dealer_hand.cards if card.rank == request]
            if dealer_matches:
                print(f"Dealer gives you: {dealer_matches}")
                self.player_hand.add_card(dealer_matches[0])
                self.dealer_hand.cards.remove(dealer_matches[0])
                self.check_pairs()
            else:
                print("Dealer says: Go Fish!")
                self.player_hand.add_card(self.deck.deal())

            if not self.dealer_hand.cards:  # If dealer runs out of cards
                print("Dealer has no more cards. You win!")
                break

    def check_pairs(self):
        for rank in set(card.rank for card in self.player_hand.cards):
            if self.player_hand.cards.count(card) == 4:
                print(f"You found a pair of {rank}!")
                self.pairs_found += 1

# Lucky Sevens Game Class
class LuckySevensGame:
    def __init__(self):
        self.deck = Deck()
        self.player_hand = Hand()

    def start_game(self):
        print("Welcome to Lucky Sevens!")
        for _ in range(3):  # Draw 3 cards
            self.player_hand.add_card(self.deck.deal())

        print("Your hand: ", self.player_hand)
        total = self.player_hand.total_value()
        print(f"Your total is: {total}")

        if total == 7:
            print("Congratulations! You've hit Lucky Sevens!")
        elif total < 7:
            print("Try again to hit Lucky Sevens.")
        else:
            print("You've gone over 7. Better luck next time!")

# Game Menu Class
class GameMenu:
    def __init__(self):
        self.games = {
            '1': BlackjackGame,
            '2': WarGame,
            '3': HigherLowerGame,
            '4': MemoryGame,
            '5': GoFishGame,
            '6': LuckySevensGame,
        }

    def display_menu(self):
        print("\nSelect a card game to play:")
        print("1. Blackjack")
        print("2. War")
        print("3. Higher or Lower")
        print("4. Memory")
        print("5. Go Fish")
        print("6. Lucky Sevens")
        choice = input("Enter the number of your choice: ")
        if choice in self.games:
            game = self.games[choice]()
            game.start_game()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu = GameMenu()
    while True:
        menu.display_menu()
        if input("Do you want to play another game? (y/n): ").lower() != 'y':
            print("Thanks for playing! Goodbye.")
            break
