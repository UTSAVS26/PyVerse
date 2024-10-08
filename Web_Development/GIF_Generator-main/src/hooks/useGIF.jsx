import React, { useEffect, useState } from 'react'
import axios from 'axios'
const API_KEY = process.env.REACT_APP_GIPHY_API_KEY;

const useGIF = (tag) => {
    const [gif, setGif] = useState('');
    const [loading, setLoading] = useState(false);
    const url = `https://api.giphy.com/v1/gifs/random?api_key=${API_KEY}&tag=${tag}`;
    console.log(url)

    async function fetchGIF() {
        setLoading(true);
        const { data } = await axios.get(url);
        const gifurl = data.data.images.downsized_large.url;
        setGif(gifurl);
        setLoading(false);
    }

    function handleGenerate() {
        fetchGIF();
    }
    useEffect(() => {
        fetchGIF();
    }, [])

  return {loading,gif,fetchGIF};
}

export default useGIF
