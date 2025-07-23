# Data structure to store progress and recommendations
anime_tracker = []
manga_tracker = []

anime_recommendations = [
    {"title": "Attack on Titan", "score": 9.1},
    {"title": "Naruto", "score": 8.2},
    {"title": "My Hero Academia", "score": 8.5},
    {"title": "Demon Slayer", "score": 8.9},
    {"title": "One Piece", "score": 9.0}
]

manga_recommendations = [
    {"title": "Berserk", "score": 9.2},
    {"title": "One Piece", "score": 8.9},
    {"title": "Tokyo Ghoul", "score": 8.5},
    {"title": "Attack on Titan", "score": 8.8},
    {"title": "My Hero Academia", "score": 8.3}
]

# Function to add anime to the tracker
def add_anime_to_tracker(anime_title, current_episode, rating):
    anime_tracker.append({
        "title": anime_title,
        "episode": current_episode,
        "rating": rating
    })
    print(f"Added {anime_title} to your anime tracker!")

# Function to add manga to the tracker
def add_manga_to_tracker(manga_title, current_chapter, rating):
    manga_tracker.append({
        "title": manga_title,
        "chapter": current_chapter,
        "rating": rating
    })
    print(f"Added {manga_title} to your manga tracker!")

# Function to show tracked anime
def show_tracked_anime():
    if anime_tracker:
        print("\nYour Tracked Anime:")
        for anime in anime_tracker:
            print(f"Title: {anime['title']}, Current Episode: {anime['episode']}, Rating: {anime['rating']}")
    else:
        print("No anime tracked yet.")

# Function to show tracked manga
def show_tracked_manga():
    if manga_tracker:
        print("\nYour Tracked Manga:")
        for manga in manga_tracker:
            print(f"Title: {manga['title']}, Current Chapter: {manga['chapter']}, Rating: {manga['rating']}")
    else:
        print("No manga tracked yet.")

# Function to recommend anime
def recommend_anime():
    print("\nAnime Recommendations:")
    for anime in anime_recommendations:
        print(f"Title: {anime['title']}, Score: {anime['score']}")

# Function to recommend manga
def recommend_manga():
    print("\nManga Recommendations:")
    for manga in manga_recommendations:
        print(f"Title: {manga['title']}, Score: {manga['score']}")

# Main function
def main():
    while True:
        print("\n--- Manga/Anime Tracker ---")
        print("1. Track Anime Progress")
        print("2. Track Manga Progress")
        print("3. Show Tracked Anime")
        print("4. Show Tracked Manga")
        print("5. Get Anime Recommendations")
        print("6. Get Manga Recommendations")
        print("7. Exit")
        choice = input("Enter your choice: ")

        match choice:
          case '1':
              title = input("Enter anime title: ")
              episode = input("Enter current episode: ")
              rating = input("Enter rating (1-10): ")
              add_anime_to_tracker(title, episode, rating)
          case '2':
              title = input("Enter manga title: ")
              chapter = input("Enter current chapter: ")
              rating = input("Enter rating (1-10): ")
              add_manga_to_tracker(title, chapter, rating)
          case '3':
              show_tracked_anime()
          case '4':
              show_tracked_manga()
          case '5':
              recommend_anime()
          case '6':
              recommend_manga()
          case '7':
              print("Exiting...")
              break
          case _:
              print("Invalid choice. Please try again.")

# Run the application
if __name__ == '__main__':
    main()