from setuptools import setup
setup(name="Playlist_downloader",
      version="1.0",
      description="Playlist Downloader simplifies the process of downloading YouTube playlists by providing a straightforward API to fetch and save videos from a given playlist URL.",
      long_description=open('README.md').read(), 
      long_description_content_type='text/markdown',
      author="Yash Kumar Saini",
      packages=['Playlist_downloader'],
      license="MIT", 
      install_requires=['pytube']
)
