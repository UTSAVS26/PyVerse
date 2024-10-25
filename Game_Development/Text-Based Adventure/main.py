def adventure_game():
    print("Welcome to the Adventure Game!")
    print("You find yourself in a dark forest. You see two paths ahead.")
    
    choice1 = input("Do you want to go left or right? (left/right): ").lower()
    
    if choice1 == "left":
        print("You walk down the left path and find a river.")
        choice2 = input("Do you want to swim across or walk along the river? (swim/walk): ").lower()

        if choice2 == "swim":
            print("You swim across the river and find a treasure chest. You win!")
        elif choice2 == "walk":
            print("You walk along the river and encounter a wild bear. You lose!")
        else:
            print("Invalid choice. You lose!")
    
    elif choice1 == "right":
        print("You walk down the right path and find a cave.")
        choice2 = input("Do you want to enter the cave or walk past it? (enter/past): ").lower()

        if choice2 == "enter":
            print("You enter the cave and find a sleeping dragon. You quietly take some gold and leave. You win!")
        elif choice2 == "past":
            print("You walk past the cave and get lost in the forest. You lose!")
        else:
            print("Invalid choice. You lose!")
    
    else:
        print("Invalid choice. You lose!")

if __name__ == "__main__":
    adventure_game()
