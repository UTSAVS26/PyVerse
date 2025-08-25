#os(operating system interface module) to work with files and folders
#json(javaScript object notation) to store the anime/manga progress in .json format
import os
import json

#Folder setup for anime/manga files to be saved(location to the folder is given )
ANIME_FOLDER = os.path.join(os.path.dirname(__file__),"Beginner_Projects", "Anime_Mange_Traker", "tracker", "anime")
os.makedirs(ANIME_FOLDER, exist_ok=True)
MANGA_FOLDER = os.path.join(os.path.dirname(__file__),"Beginner_Projects", "Anime_Mange_Traker", "tracker", "manga")
os.makedirs(MANGA_FOLDER, exist_ok=True)

#anime recommendation dictionary
anime_recommendations = [
    {"title": "Attack on Titan", "score": 9.1},
    {"title": "Naruto", "score": 8.2},
    {"title": "My Hero Academia", "score": 8.5},
    {"title": "Demon Slayer", "score": 8.9},
    {"title": "One Piece", "score": 9.0}
]

#manga recommendation dictionary
manga_recommendations = [
    {"title": "Berserk", "score": 9.2},
    {"title": "One Piece", "score": 8.9},
    {"title": "Tokyo Ghoul", "score": 8.5},
    {"title": "Attack on Titan", "score": 8.8},
    {"title": "My Hero Academia", "score": 8.3}
]

#function for getting ratings for anime/manga input
def get_rating_input():
    """Ensure rating is between 1 and 10."""
    while True:
        try:
            rating = float(input("Enter rating (1–10): "))
            if 1 <= rating <= 10:
                return rating
            else:
                print("❌ Rating must be between 1 and 10.")
        except ValueError:
            print("❌ Please enter a number.")

#funtion for storing anime progress in .json file
def add_anime_progress(anime_title, episode_num, rating):
    

    file_name = anime_title.replace(" ", "_") + ".json"
    file_path = os.path.join(ANIME_FOLDER, file_name)

    #load existing file if it exist otherwise create one for storing the data
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            anime_data = json.load(f)
    else:
        anime_data = {"title": anime_title, "episodes": [], "overall_rating": 0}

    #add new episode or rating
    anime_data["episodes"].append({"number": episode_num, "rating": rating})

    #recalculate overall ratings
    ratings = [float(ep["rating"]) for ep in anime_data["episodes"]]
    anime_data["overall_rating"] = round(sum(ratings) / len(ratings), 2)

    #save the updated file back to the file
    with open(file_path, "w") as f:
        json.dump(anime_data, f, indent=4)

    print(f"✅ Progress saved for {anime_title}! Overall rating: {anime_data['overall_rating']}")

#function for storing manga data in .json file
def add_manga_progress(manga_title, current_chapter, rating):
    file_name = manga_title.replace(" ", "_") + ".json"
    file_path = os.path.join(MANGA_FOLDER, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            manga_data = json.load(f)
    else:
        manga_data = {"title": manga_title, "chapters": [], "overall_rating": 0}

    manga_data["chapters"].append({"number": current_chapter, "rating": rating})

    ratings = [float(chap["rating"]) for chap in manga_data["chapters"]]
    manga_data["overall_rating"] = round(sum(ratings) / len(ratings), 2)

    with open(file_path, "w") as f:
        json.dump(manga_data, f, indent=4)

    print(f"✅ Progress saved for {manga_title}! Overall rating: {manga_data['overall_rating']}")

#function which fetch data from the respective anime .json file and show the elements in the output
def view_anime_progress(anime_title):

    # Create a valid file name by replacing spaces with underscores    file_name = anime_title.replace(" ", "_") + ".json"
    file_name = anime_title.replace(" ", "_") + ".json"
    # Build the full path to the JSON file inside the ANIME_FOLDER
    file_path = os.path.join(ANIME_FOLDER, file_name)

    #if the entered file_name does not exist, inform the user and strop the function
    if not os.path.exists(file_path):
        print("⚠ No progress found for this anime.")
        return

    with open(file_path, "r") as f:
        #load the json elements into a python dictionary
        anime_data = json.load(f)

    print(f"\n📺 {anime_data['title']}")
    for ep in anime_data["episodes"]:
        print(f"Episode {ep['number']}: Rating {ep['rating']}")
    print(f"⭐ Overall Rating: {anime_data['overall_rating']}")

#function which fetch data from the respective manga .json file and show the elements in the output
def view_manga_progress(manga_title):
    file_name = manga_title.replace(" ", "_") + ".json"
    file_path = os.path.join(MANGA_FOLDER, file_name)

    if not os.path.exists(file_path):
        print("⚠ No progress found for this manga.")
        return

    with open(file_path, "r") as f:
        manga_data = json.load(f)

    print(f"\n📚 {manga_data['title']}")
    for chap in manga_data["chapters"]:
        print(f"Chapter {chap['number']}: Rating {chap['rating']}")
    print(f"⭐ Overall Rating: {manga_data['overall_rating']}")

#functions to recommend anime and manga
def recommend_anime():
    print("\n📺 Anime Recommendations:")
    for anime in anime_recommendations:
        print(f"{anime['title']} — Score: {anime['score']}")

def recommend_manga():
    print("\n📚 Manga Recommendations:")
    for manga in manga_recommendations:
        print(f"{manga['title']} — Score: {manga['score']}")

#Main function
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

        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
            continue

        match choice:
            case 1:
                title = input("Enter anime title: ")
                episode = int(input("Enter current episode: "))
                rating = get_rating_input()
                add_anime_progress(title, episode, rating)
            case 2:
                title = input("Enter manga title: ")
                chapter = int(input("Enter current chapter: "))
                rating = get_rating_input()
                add_manga_progress(title, chapter, rating)
            case 3:
                title = input("Enter the anime title to view progress: ")
                view_anime_progress(title)
            case 4:
                title = input("Enter the manga title to view progress: ")
                view_manga_progress(title)
            case 5:
                recommend_anime()
            case 6:
                recommend_manga()
            case 7:
                print("👋 Exiting...")
                break
            case _:
                print("❌ Invalid choice. Please try again.")

#Run the application
if __name__ == "__main__":
    main()
