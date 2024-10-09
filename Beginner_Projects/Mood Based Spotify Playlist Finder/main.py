import spotipy
from redis.commands.search.result import Result
from spotipy.oauth2 import SpotifyOAuth


def song_call(user_mood):
    try:
        scope = "user-library-read"
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
    except Exception as e:
        print("Connection Failed", e)

    try:
        result = sp.search(q=user_mood + "genre", type="playlist")
        print(result['playlists']['items'][0]['external_urls']['spotify'])
        results = sp.playlist_items(result['playlists']['items'][0]['external_urls']['spotify'])
        for idx, item in enumerate(results['items']):
            track = item['track']

            print(idx + 1, track['artists'][0]['name'], " – ", track['name'])
    except Exception as e:
        print("Searching Error", e)


def main():
    moods_available = ["fun","sad","happy","romantic","adventurous","entertainment"]
    while True:
        user_mood = input("What are you Currently Feeling ;)")
        if user_mood.lower() in moods_available:
            break
        else:
            print("Please enter a valid mood!")
    song_call(user_mood)


if __name__ == '__main__':
    main()
