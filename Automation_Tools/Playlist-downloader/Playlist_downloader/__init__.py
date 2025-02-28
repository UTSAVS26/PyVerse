from pytube import Playlist
def download(playlist_id,download_path):
    try:
        playlist_url = 'https://www.youtube.com/playlist?list=' +playlist_id
        # Create a Playlist object
        playlist = Playlist(playlist_url)
        # download_path=r'{}'.format(download_path)
        # Specify the directory where you want to save the downloaded videos
        # Iterate through each video in the playlist and download it to the specified directory
        for video in playlist.videos:
            try:
                print(f'Downloading: {video.title}')
                video.streams.get_highest_resolution().download(output_path=download_path)
                print(f'{video.title} downloaded successfully!')
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
