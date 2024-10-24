import turtle
import pandas

screen = turtle.Screen()
screen.title("Guess The Indian State")
screen.setup(width = 650, height = 700)
screen.addshape("PyVerse/Game_Development/Map Game/Indian_Map.gif")
turtle.shape("PyVerse/Game_Development/Map Game/Indian_Map.gif")

location = pandas.read_csv("PyVerse/Game_Development/Map Game/Indian_states.csv")
all_states = location["State"].to_list()

def map_game():
    guess = screen.textinput("Guess The State","Enter The Name of the Next State ?")
    if guess: 
        guess = guess.title()
        
        score = 0

        flag = True
        while flag:
            if guess in all_states:
                state_name = turtle.Turtle()
                state_name.hideturtle()
                state_name.penup()
                new_location = location[location.State == guess]
                score += 1
                state_name.goto(x = new_location["Latitude"].item(), y = new_location["Longitude"].item())
                state_name.write(new_location["State"].item())
                all_states.remove(guess)
            elif guess == "Exit":
                print (all_states)
                print (len(all_states))
                flag = False
            guess = screen.textinput(f"{score}/29 Guessed Correctly","Enter The Name of the Next State ?")
            if guess: 
                guess = guess.title()
                continue
            else:
                print(f"Quiting The Game with Score {score}/29....")
                flag =  False
                turtle.bye()
            if score > 28:
                flag = False
                turtle.done()
                
    else:
        print("Quiting The Game....")

    
if __name__ == "__main__":
    map_game()