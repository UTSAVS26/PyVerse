import random

def add_song(songs):
    title = input("Enter song title: ")
    artist = input("Enter artist: ")
    genre = input("Enter genre: ")
    mood = input("Enter mood: ")
    songs[title] = {'artist': artist, 'genre': genre, 'mood': mood}
    print(f"✅ '{title}' added!")

def view_songs(songs):
    if not songs:
        print("No songs available.")
        return
    print("\n🎵 All Songs:")
    for title, info in songs.items():
        print(f"- {title} | {info['artist']} | {info['genre']} | {info['mood']}")

def search_songs(songs, key, value):
    found = [title for title, info in songs.items() if info[key].lower() == value.lower()]
    if found:
        print(f"\n🔍 Found songs for {key} '{value}':")
        for song in found:
            print(f"- {song}")
    else:
        print(f"No songs found for {key} '{value}'.")

def create_playlist(songs, playlists):
    name = input("Enter playlist name: ")
    playlist = []
    while True:
        title = input("Add song to playlist (or type 'done'): ")
        if title.lower() == 'done':
            break
        if title in songs:
            playlist.append(title)
        else:
            print("❌ Song not found.")
    playlists[name] = playlist
    print(f"🎧 Playlist '{name}' created with {len(playlist)} songs!")

def view_playlist(playlists):
    name = input("Enter playlist name to view: ")
    if name in playlists:
        print(f"\n📃 Playlist: {name}")
        for song in playlists[name]:
            print(f"- {song}")
    else:
        print("❌ Playlist not found.")

def random_mood_playlist(songs):
    mood = input("Enter mood: ")
    matching = [title for title, info in songs.items() if info['mood'].lower() == mood.lower()]
    if matching:
        random.shuffle(matching)
        print(f"\n🎲 Random Playlist for mood '{mood}':")
        for song in matching:
            print(f"- {song}")
    else:
        print("❌ No songs match that mood.")

def main():
    songs = {
        'Shape of You': {'artist': 'Ed Sheeran', 'genre': 'Pop', 'mood': 'happy'},
        'Blinding Lights': {'artist': 'The Weeknd', 'genre': 'Synthpop', 'mood': 'energetic'},
        'Someone Like You': {'artist': 'Adele', 'genre': 'Ballad', 'mood': 'sad'},
        'Believer': {'artist': 'Imagine Dragons', 'genre': 'Rock', 'mood': 'energetic'},
        'Perfect': {'artist': 'Ed Sheeran', 'genre': 'Pop', 'mood': 'romantic'},
        'Let Her Go': {'artist': 'Passenger', 'genre': 'Folk', 'mood': 'sad'},
        'Happy': {'artist': 'Pharrell Williams', 'genre': 'Pop', 'mood': 'happy'},
        'Thunder': {'artist': 'Imagine Dragons', 'genre': 'Rock', 'mood': 'motivational'},
        'Senorita': {'artist': 'Shawn Mendes', 'genre': 'Latin Pop', 'mood': 'romantic'},
        'Dance Monkey': {'artist': 'Tones and I', 'genre': 'Alternative', 'mood': 'energetic'},
        'Lose Yourself': {'artist': 'Eminem', 'genre': 'Hip-Hop', 'mood': 'motivational'},
        'Lovely': {'artist': 'Billie Eilish', 'genre': 'Alternative', 'mood': 'sad'},
        'Counting Stars': {'artist': 'OneRepublic', 'genre': 'Pop Rock', 'mood': 'uplifting'},
        'Stay': {'artist': 'Justin Bieber', 'genre': 'Pop', 'mood': 'romantic'},
        'Cheap Thrills': {'artist': 'Sia', 'genre': 'Pop', 'mood': 'party'},
        'Radioactive': {'artist': 'Imagine Dragons', 'genre': 'Rock', 'mood': 'intense'},
        'Let Me Love You': {'artist': 'DJ Snake', 'genre': 'Electronic', 'mood': 'romantic'},
        'Heat Waves': {'artist': 'Glass Animals', 'genre': 'Indie', 'mood': 'nostalgic'},
        'Peaches': {'artist': 'Justin Bieber', 'genre': 'Pop', 'mood': 'chill'},
        'Night Changes': {'artist': 'One Direction', 'genre': 'Pop Rock', 'mood': 'soft'},
        'Halo': {'artist': 'Beyoncé', 'genre': 'R&B', 'mood': 'inspirational'},
        'Levitating': {'artist': 'Dua Lipa', 'genre': 'Disco Pop', 'mood': 'happy'},
        'Faded': {'artist': 'Alan Walker', 'genre': 'Electronic', 'mood': 'melancholic'},
        'Closer': {'artist': 'The Chainsmokers', 'genre': 'EDM', 'mood': 'romantic'},
        'Stressed Out': {'artist': 'Twenty One Pilots', 'genre': 'Alternative', 'mood': 'relatable'},
        'Heat Waves': {'artist': 'Glass Animals', 'genre': 'Indie Pop', 'mood': 'nostalgic'},
        'Levitating': {'artist': 'Dua Lipa', 'genre': 'Pop', 'mood': 'energetic'},
        'Shivers': {'artist': 'Ed Sheeran', 'genre': 'Pop Rock', 'mood': 'romantic'},
        'Blinding Lights': {'artist': 'The Weeknd', 'genre': 'Synthwave', 'mood': 'excited'},
        'Bad Habit': {'artist': 'Steve Lacy', 'genre': 'Alternative R&B', 'mood': 'moody'},
        'About Damn Time': {'artist': 'Lizzo', 'genre': 'Funk Pop', 'mood': 'confident'},
        'Golden Hour': {'artist': 'JVKE', 'genre': 'Pop', 'mood': 'dreamy'},
        'As It Was': {'artist': 'Harry Styles', 'genre': 'Pop Rock', 'mood': 'reflective'},
        'Stay': {'artist': 'The Kid LAROI & Justin Bieber', 'genre': 'Pop', 'mood': 'emotional'},
        'good 4 u': {'artist': 'Olivia Rodrigo', 'genre': 'Pop Punk', 'mood': 'angry'}, 
        'Bohemian Rhapsody': {'artist': 'Queen', 'genre': 'Rock', 'mood': 'dramatic'},
        'Sunflower': {'artist': 'Post Malone & Swae Lee', 'genre': 'Hip Hop', 'mood': 'chill'},
        'Someone Like You': {'artist': 'Adele', 'genre': 'Soul', 'mood': 'heartbroken'},
        "Can't Stop the Feeling!": {'artist': 'Justin Timberlake', 'genre': 'Pop', 'mood': 'joyful'},
        'Believer': {'artist': 'Imagine Dragons', 'genre': 'Alternative Rock', 'mood': 'intense'},
        'Lucid Dreams': {'artist': 'Juice WRLD', 'genre': 'Emo Rap', 'mood': 'melancholy'},
        'Uptown Funk': {'artist': 'Mark Ronson ft. Bruno Mars', 'genre': 'Funk', 'mood': 'energetic'},
        'Counting Stars': {'artist': 'OneRepublic', 'genre': 'Pop Rock', 'mood': 'motivated'},
        'Lovely': {'artist': 'Billie Eilish & Khalid', 'genre': 'Indie Pop', 'mood': 'haunting'},
        'Faded': {'artist': 'Alan Walker', 'genre': 'Electronic', 'mood': 'mysterious'},
        'Thunder': {'artist': 'Imagine Dragons', 'genre': 'Electronic Rock', 'mood': 'bold'},
        'Peaches': {'artist': 'Justin Bieber ft. Daniel Caesar & Giveon', 'genre': 'R&B', 'mood': 'romantic'},
        'Take Five': {'artist': 'The Dave Brubeck Quartet', 'genre': 'Jazz', 'mood': 'cool'},
        'Despacito': {'artist': 'Luis Fonsi ft. Daddy Yankee', 'genre': 'Reggaeton', 'mood': 'vibrant'},
        'Waka Waka': {'artist': 'Shakira', 'genre': 'World Music', 'mood': 'festive'},
        'Sweater Weather': {'artist': 'The Neighbourhood', 'genre': 'Indie Rock', 'mood': 'moody'},
        'Dance Monkey': {'artist': 'Tones and I', 'genre': 'Electropop', 'mood': 'quirky'},
        'Happier Than Ever': {'artist': 'Billie Eilish', 'genre': 'Alternative', 'mood': 'empowered'},
        'Lose Yourself': {'artist': 'Eminem', 'genre': 'Rap', 'mood': 'determined'},
        'Photograph': {'artist': 'Ed Sheeran', 'genre': 'Acoustic', 'mood': 'sentimental'}, 
        'Murder in My Mind': {'artist': 'Kordhell', 'genre': 'Phonk', 'mood': 'dark'},
        'GigaChad Theme (Phonk Remix)': {'artist': 'Phonk Killer', 'genre': 'Phonk', 'mood': 'powerful'},
        'NEVER FORGIVE': {'artist': 'KSLV Noh', 'genre': 'Phonk', 'mood': 'aggressive'},
        'CHAOS': {'artist': 'Kordhell', 'genre': 'Phonk', 'mood': 'intense'},
        'PHONKY TOWN': {'artist': 'PlayaPhonk', 'genre': 'Phonk', 'mood': 'retro'},
        'Midnight': {'artist': 'lxst cxntury', 'genre': 'Phonk', 'mood': 'mysterious'},
        'DRIFT PHONK': {'artist': 'RYO', 'genre': 'Phonk', 'mood': 'adrenaline'},
        'TRAP PHONK': {'artist': 'MoonDeity', 'genre': 'Phonk', 'mood': 'hard-hitting'},
        'RAVE': {'artist': 'Kordhell', 'genre': 'Phonk', 'mood': 'hyped'},
        'SHE': {'artist': 'Interworld', 'genre': 'Phonk', 'mood': 'melancholy'}
    }

    playlists = {}

    while True:
        print("\n🎶 Playlist Organizer Menu")
        print("1. Add a Song")
        print("2. View All Songs")
        print("3. Search Songs by Genre")
        print("4. Search Songs by Mood")
        print("5. Create a Playlist")
        print("6. View a Playlist")
        print("7. Generate Random Playlist by Mood")
        print("8. Exit")

        choice = input("Choose an option (1-8): ")

        if choice == '1':
            add_song(songs)
        elif choice == '2':
            view_songs(songs)
        elif choice == '3':
            genre = input("Enter genre to search: ")
            search_songs(songs, 'genre', genre)
        elif choice == '4':
            mood = input("Enter mood to search: ")
            search_songs(songs, 'mood', mood)
        elif choice == '5':
            create_playlist(songs, playlists)
        elif choice == '6':
            view_playlist(playlists)
        elif choice == '7':
            random_mood_playlist(songs)
        elif choice == '8':
            print("👋 Goodbye! Enjoy your music.")
            break
        else:
            print("❌ Invalid option. Try again.")

if __name__ == '__main__':
    main()
