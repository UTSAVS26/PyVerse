def main_menu():
    print("Welcome to the Visual Novel!")
    print("Which adventure do you want to dive into?")
    print("1. The Lost Treasure")
    print("2. The Haunted House")
    print("3. The Secret Garden")
    print("4. The Time Traveler")
    print("5. The Mysterious Stranger")
    
    choice = input("Please enter the number of your choice: ")
    
    if choice == '1':
        lost_treasure()
    elif choice == '2':
        haunted_house()
    elif choice == '3':
        secret_garden()
    elif choice == '4':
        time_traveler()
    elif choice == '5':
        mysterious_stranger()
    else:
        print("Hmm, that doesn't seem right. Please try again.")
        main_menu()

def lost_treasure():
    print("\nYou're on an adventure to find the legendary treasure of Captain Flint.")
    print("You stand at the entrance of a dark cave where the treasure is said to be hidden.")
    choice = input("Do you want to venture inside the cave (yes/no)? ")
    
    if choice.lower() == 'yes':
        print("Inside, you discover an ancient map and hear a voice warning you about traps.")
        choice = input("Do you want to follow the map (yes) or check for traps first (no)? ")
        
        if choice.lower() == 'yes':
            print("You follow the map and suddenly, a swinging axe trap almost catches you!")
            choice = input("Do you want to try to disarm it (yes) or jump back (no)? ")
            if choice.lower() == 'yes':
                print("You skillfully disarm the trap and find a treasure chest overflowing with gold!")
                choice = input("Do you want to open the chest (yes) or leave it closed (no)? ")
                if choice.lower() == 'yes':
                    print("Inside, you find not just gold, but a mysterious cursed artifact!")
                    choice = input("Do you want to keep the artifact (yes) or leave it behind (no)? ")
                    if choice.lower() == 'yes':
                        print("You take the artifact, but soon find it brings you nothing but bad luck.")
                    else:
                        print("You wisely leave the artifact behind and head home safely.")
                else:
                    print("You decide to leave the chest closed and escape the cave with just the map.")
            else:
                print("You jump back just in time, losing your chance at treasure.")
        
        else:
            print("You find an old pirate's diary. It has clues and warns of a rival treasure hunter!")
            choice = input("Do you want to decipher the diary (yes) or ignore it and leave (no)? ")
            if choice.lower() == 'yes':
                print("The diary leads you to a hidden treasure spot, but you realize the rival is close!")
                choice = input("Do you want to set a trap for the rival (yes) or confront them (no)? ")
                if choice.lower() == 'yes':
                    print("You successfully trap the rival and claim the treasure for yourself!")
                else:
                    print("You confront them, and after a tense moment, you both agree to share the treasure.")
            else:
                print("You leave the diary behind, and the rival finds it first, claiming the treasure.")
    
    else:
        print("You decide it's not worth the risk and leave the cave. Maybe another adventure awaits.")

def haunted_house():
    print("\nYou find yourself drawn to a haunted house known for its restless spirits.")
    print("As you step inside, the door creaks ominously behind you.")
    choice = input("Do you want to go upstairs (yes) or check out the basement (no)? ")
    
    if choice.lower() == 'yes':
        print("Upstairs, you meet a ghost who seems troubled.")
        choice = input("Do you want to help the ghost find peace (yes) or run away (no)? ")
        if choice.lower() == 'yes':
            print("The ghost tells you there's treasure hidden in the attic!")
            choice = input("Do you want to take the treasure (yes) or leave it for the ghost (no)? ")
            if choice.lower() == 'yes':
                print("You take the treasure, but the ghost curses you with bad luck!")
                choice = input("Do you want to try to lift the curse (yes) or accept your fate (no)? ")
                if choice.lower() == 'yes':
                    print("You find a way to break the curse and live a fortunate life.")
                else:
                    print("You live with the bad luck for years, always haunted by the ghost.")
            else:
                print("You leave the treasure, and in gratitude, the ghost grants you a wish!")
        else:
            print("You run away, but the ghost follows, seeking revenge.")
    
    else:
        print("In the basement, you discover a hidden room filled with strange artifacts and a glowing portal.")
        choice = input("Do you want to enter the portal (yes) or check out the artifacts (no)? ")
        if choice.lower() == 'yes':
            print("You find yourself in a magical realm filled with fantastical creatures!")
            choice = input("Do you want to explore this world (yes) or go back to the house (no)? ")
            if choice.lower() == 'yes':
                print("You make friends with a magical creature who offers to guide you!")
                choice = input("Do you want to accept the creature's offer (yes) or venture alone (no)? ")
                if choice.lower() == 'yes':
                    print("Together, you uncover ancient secrets and hidden treasures.")
                else:
                    print("You face dangers alone but still find hidden treasures.")
            else:
                print("You return to the basement, but the portal has closed behind you.")
        else:
            print("You uncover a diary that reveals the house's dark history.")
            choice = input("Do you want to read the diary (yes) or leave it (no)? ")
            if choice.lower() == 'yes':
                print("The diary tells of the ghost's tragic past, and you feel compelled to help.")
                choice = input("Do you want to help the ghost (yes) or leave the house (no)? ")
                if choice.lower() == 'yes':
                    print("You help the ghost find peace, and it rewards you with a magical gift!")
                else:
                    print("You leave, feeling a heavy sense of foreboding.")

def secret_garden():
    print("\nYou stumble upon a hidden garden behind a tall hedge.")
    print("The flowers glow mysteriously under the moonlight.")
    choice = input("Do you want to pick a flower (yes) or explore the garden (no)? ")
    
    if choice.lower() == 'yes':
        print("As you pick a flower, magical creatures appear and surround you.")
        choice = input("Do you want to talk to them (yes) or run away (no)? ")
        if choice.lower() == 'yes':
            print("They grant you a wish for your bravery!")
            choice = input("What do you wish for? (adventure/money/love): ")
            if choice.lower() == 'adventure':
                print("You’re whisked away on a thrilling quest filled with challenges!")
                choice = input("Do you want to fight a dragon (yes) or seek hidden treasure (no)? ")
                if choice.lower() == 'yes':
                    print("You bravely face the dragon and earn its respect!")
                else:
                    print("You find a treasure map leading to even more adventures!")
            elif choice.lower() == 'money':
                print("You receive a treasure chest filled with gold coins!")
            else:
                print("You find your true love waiting for you in the garden!")
        else:
            print("You drop the flower and escape, your heart racing.")
    
    else:
        print("You wander deeper and discover a hidden pond reflecting your image.")
        choice = input("Do you want to touch the water (yes) or leave (no)? ")
        if choice.lower() == 'yes':
            print("You catch glimpses of possible futures in the water!")
            choice = input("Do you want to stay and watch (yes) or leave (no)? ")
            if choice.lower() == 'yes':
                print("You're mesmerized and lose track of time!")
            else:
                print("You decide to leave, feeling a mix of wonder and curiosity.")
        else:
            print("You find a hidden path leading to a secret gate.")
            choice = input("Do you want to open the gate (yes) or ignore it (no)? ")
            if choice.lower() == 'yes':
                print("You discover a hidden world beyond!")
                choice = input("Do you want to explore this world (yes) or return (no)? ")
                if choice.lower() == 'yes':
                    print("You encounter strange creatures and forge new friendships!")
                else:
                    print("You walk away, leaving the mystery behind.")
            else:
                print("You walk away, but the garden stays in your mind.")

def time_traveler():
    print("\nYou come across a strange device that looks like a time machine.")
    print("Do you want to give it a try?")
    choice = input("Enter (yes/no): ")
    
    if choice.lower() == 'yes':
        print("You activate the device and find yourself in the future!")
        choice = input("Do you want to ask your future self about your life (yes) or explore the future (no)? ")
        if choice.lower() == 'yes':
            print("Your future self shares valuable advice that could change your present!")
            choice = input("Do you want to follow this advice (yes) or ignore it (no)? ")
            if choice.lower() == 'yes':
                print("You lead a happier life and achieve your dreams!")
            else:
                print("You ignore the advice, leading to unexpected challenges.")
        else:
            print("You wander a world of flying cars and advanced technology.")
            choice = input("Do you want to stay here (yes) or return to your time (no)? ")
            if choice.lower() == 'yes':
                print("You decide to start fresh in the future, becoming a trailblazer!")
            else:
                print("You return home, eager to share your incredible stories.")
    
    else:
        print("You leave the device untouched, but it continues to intrigue you.")
        choice = input("Do you want to investigate the device later (yes) or forget about it (no)? ")
        if choice.lower() == 'yes':
            print("You return and accidentally activate it, launching into time!")
            choice = input("Do you want to explore the past or the future (past/future)? ")
            if choice.lower() == 'past':
                print("You witness historical events but must be careful not to change history!")
                choice = input("Do you want to talk to a historical figure (yes) or stay hidden (no)? ")
                if choice.lower() == 'yes':
                    print("You have a brief conversation that changes how you view your life.")
                else:
                    print("You remain hidden and observe a pivotal moment in history.")
            else:
                print("You see a future that isn’t what you expected!")
                choice = input("Do you want to embrace this future (yes) or change your path (no)? ")
                if choice.lower() == 'yes':
                    print("You accept your fate and thrive in your new reality!")
                else:
                    print("You seek a way back to your present, fighting against destiny.")
        else:
            print("You forget about it, but it lingers in your thoughts.")

def mysterious_stranger():
    print("\nA mysterious stranger approaches you in a bustling market.")
    print("They whisper secrets of the universe, hinting at hidden powers.")
    choice = input("Do you want to follow them (yes) or ignore them (no)? ")
    
    if choice.lower() == 'yes':
        print("You find yourself on a journey of knowledge and discovery!")
        choice = input("Do you want to learn their secrets (yes) or ask about your future (no)? ")
        if choice.lower() == 'yes':
            print("You gain insights that change how you see the world!")
            choice = input("Do you want to share this knowledge (yes) or keep it to yourself (no)? ")
            if choice.lower() == 'yes':
                print("You inspire others and become a respected figure!")
            else:
                print("You hold onto the knowledge, feeling both its weight and responsibility.")
        else:
            print("They show you a glimpse of your future, revealing two possible paths.")
            choice = input("Do you want to pursue adventure (adventure) or wisdom (wisdom)? ")
            if choice.lower() == 'adventure':
                print("You face challenges that help you discover your true strength!")
                choice = input("Do you want to take on a dangerous quest (yes) or gather allies first (no)? ")
                if choice.lower() == 'yes':
                    print("You prove your bravery and gain the respect of many!")
                else:
                    print("With friends by your side, you tackle the challenges together!")
            else:
                print("You choose to seek knowledge and uncover ancient mysteries.")
                choice = input("Do you want to share these discoveries (yes) or keep them secret (no)? ")
                if choice.lower() == 'yes':
                    print("You become a renowned scholar and mentor to many.")
                else:
                    print("You safeguard the secrets, forever pondering their implications.")
    
    else:
        print("You walk away, wondering what might have been.")
        choice = input("Do you want to seek out the stranger again (yes) or move on (no)? ")
        if choice.lower() == 'yes':
            print("You find them again, and they offer you a choice: power or wisdom.")
            choice = input("Which do you choose (power/wisdom)? ")
            if choice.lower() == 'power':
                print("You gain incredible abilities, but at a significant personal cost.")
            else:
                print("You become a wise sage, though some power remains just out of reach.")
        else:
            print("You continue on your path, but the mystery of the stranger lingers in your thoughts.")

if __name__ == "__main__":
    main_menu()
