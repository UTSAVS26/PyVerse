import random

# Expanded playlists with moods and languages
playlists = {
    "happy": {
        "english": [
            "Happy - Pharrell Williams", "Uptown Funk - Bruno Mars", "Can't Stop the Feeling - Justin Timberlake",
            "Shake It Off - Taylor Swift", "Best Day of My Life - American Authors"
        ],
        "hindi": [
            "Gallan Goodiyan - Dil Dhadakne Do", "Subah Hone Na De - Desi Boyz", "London Thumakda - Queen",
            "Balam Pichkari - Yeh Jawaani Hai Deewani", "Cutiepie - Ae Dil Hai Mushkil"
        ],
        "kpop": [
            "Dynamite - BTS", "Lovesick Girls - BLACKPINK", "Psycho - Red Velvet", 
            "LALISA - Lisa", "Ice Cream - BLACKPINK ft. Selena Gomez"
        ],
        "tamil": [
            "Vaathi Coming - Master", "Megham Karukatha - Thiruchitrambalam", 
            "Why This Kolaveri Di - 3", "Jolly O Gymkhana - Beast", "Aaluma Doluma - Vedalam"
        ]
    },
    "sad": {
        "english": [
            "Someone Like You - Adele", "Fix You - Coldplay", "Let Her Go - Passenger",
            "Stay With Me - Sam Smith", "All I Want - Kodaline"
        ],
        "hindi": [
            "Tujhe Bhula Diya - Anjaana Anjaani", "Tum Hi Ho - Aashiqui 2", "Channa Mereya - Ae Dil Hai Mushkil",
            "Agar Tum Saath Ho - Tamasha", "Phir Le Aya Dil - Barfi"
        ],
        "kpop": [
            "Spring Day - BTS", "Stay - BLACKPINK", "Hold On - NCT 127", 
            "Love Poem - IU", "Blue & Grey - BTS"
        ],
        "tamil": [
            "Ennodu Nee Irundhal - I", "Oru Deivam Thantha Poove - Kannathil Muthamittal", 
            "Nenjukkul Peidhidum - Vaaranam Aayiram", "Uyire - Bombay", "Kanmani Anbodu - Guna"
        ]
    },
    "relaxed": {
        "english": [
            "Weightless - Marconi Union", "Clair de Lune - Debussy", "Chill Vibes - Various Artists",
            "Easy On Me - Adele", "The Night We Met - Lord Huron"
        ],
        "hindi": [
            "Tum Mile - Tum Mile", "Phir Se Ud Chala - Rockstar", "Dil Dhadakne Do - ZNMD",
            "Ilahi - Yeh Jawaani Hai Deewani", "Pee Loon - Once Upon a Time in Mumbaai"
        ],
        "kpop": [
            "Palette - IU ft. G-Dragon", "Eight - IU ft. Suga", "Our Summer - TXT", 
            "Love Scenario - iKON", "Serendipity - BTS"
        ],
        "tamil": [
            "Munbe Vaa - Sillunu Oru Kadhal", "New York Nagaram - Sillunu Oru Kadhal", 
            "Nenjukulle - Kadal", "Maruvaarthai - Enai Noki Paayum Thota", "Kangal Irandal - Subramaniapuram"
        ]
    },
    "energetic": {
        "english": [
            "Eye of the Tiger - Survivor", "Stronger - Kanye West", "Don't Stop Me Now - Queen",
            "We Will Rock You - Queen", "Thunder - Imagine Dragons"
        ],
        "hindi": [
            "Kala Chashma - Baar Baar Dekho", "Kar Gayi Chull - Kapoor & Sons", 
            "Zingaat - Dhadak", "Saturday Saturday - Humpty Sharma Ki Dulhania", "Bang Bang - Bang Bang"
        ],
        "kpop": [
            "Mic Drop - BTS", "Fire - BTS", "Kill This Love - BLACKPINK", 
            "I'm the Best - 2NE1", "Warrior - B.A.P"
        ],
        "tamil": [
            "Rakita Rakita - Jagame Thandhiram", "Kutti Story - Master", "Vaadi Vaadi - Sachein", 
            "Dandanakka - Romeo Juliet", "Anirudh Mashup - Anirudh Ravichander"
        ]
    }
}

# Welcome message
print("🎵 Welcome to the Mood-Based Playlist Generator! 🎵\n")

# Ask the user for their mood
print("Moods available: Happy, Sad, Relaxed, Energetic")
mood = input("How are you feeling today? (e.g., happy, sad, relaxed, energetic): ").strip().lower()

# Ask the user for their language preference
print("\nLanguages available: English, Hindi, K-pop, Tamil")
language = input("Which language do you prefer? (e.g., english, hindi, kpop, tamil): ").strip().lower()

# Generate and display the playlist
if mood in playlists and language in playlists[mood]:
    print(f"\nHere's a {language.capitalize()} playlist to match your mood ({mood.capitalize()}):\n")
    songs = playlists[mood][language]
    # Display 5 random songs from the playlist
    for song in random.sample(songs, min(5, len(songs))):
        print(f"- {song}")
else:
    print("\nSorry, I don't have a playlist for that combination yet. Try another mood or language!")

print("\nThank you for using the Mood-Based Playlist Generator! 🎧")
