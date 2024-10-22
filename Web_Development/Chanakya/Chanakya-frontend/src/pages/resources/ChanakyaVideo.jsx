import React, { useEffect, useState } from 'react';
import axios from 'axios';

const API_KEY = import.meta.env.VITE_YOUTUBE_API_KEY;
const SEARCH_QUERY = 'chanakya niti, chanakya, story of chanakya, chanakya life';
const MAX_RESULTS = 9;

const ChanakyaVideo = () => {
  const [videos, setVideos] = useState([]);
  const [visibleVideos, setVisibleVideos] = useState([]);
  const [nextPageToken, setNextPageToken] = useState('');

  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    try {
      const response = await axios.get(
        `https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=${MAX_RESULTS}&q=${SEARCH_QUERY}&regionCode=IN&type=video&key=${API_KEY}&pageToken=${nextPageToken}`
      );
      const fetchedVideos = response.data.items;
      setVideos(prevVideos => [...prevVideos, ...fetchedVideos]);
      setNextPageToken(response.data.nextPageToken);
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

  const handleLoadMore = () => {
    fetchVideos();
  };

  useEffect(() => {
    setVisibleVideos(videos.slice(0, visibleVideos.length + 3));
  }, [videos, nextPageToken]);
  const rows = Math.ceil(visibleVideos.length / 3);

  return (
    <div className="container py-4">
      <h1 className="text-center mb-4">Chanakya Niti & Chanakya Videos</h1>
      <div className="row row-cols-1 row-cols-md-3 g-4">
        {visibleVideos.map((video) => (
          <div key={video.id.videoId} className="col">
            <div className="card h-100 shadow-sm w-full">
              <iframe
                title={video.snippet.title}
                src={`https://www.youtube.com/embed/${video.id.videoId}`}
                className="card-img-top"
                style={{ border: 'none', height: '100%' }}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              ></iframe>
              <div className="card-body">
                <h2 className="card-title h5">{video.snippet.title}</h2>
              </div>
            </div>
          </div>
        ))}
      </div>
      {rows * 3 < videos.length && (
        <div className="d-flex justify-content-center mt-4">
          <button className="btn btn-primary" onClick={handleLoadMore}>
            Load More
          </button>
        </div>
      )}
    </div>
  );
};

export default ChanakyaVideo;
