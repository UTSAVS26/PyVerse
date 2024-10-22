import React, { useContext, useEffect, useState } from "react";
import * as func from '../../functions/RequestEpisode.module';
import ReactPlayer from "react-player";
import { Context } from "../../context/Context";

const ChanakyaAudio = () => {
  const { setProgress } = useContext(Context);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [URL, setURL] = useState(null);
  const [episodeNumber, setEpisodeNumber] = useState(1);
  const [value, setValue] = useState(1);
  const [playbackRate, setPlaybackRate] = useState(1);

  const changePlaybackRate = (rate) => {
    setPlaybackRate(rate);
  };

  useEffect(() => {
    const getData = async () => {
      setProgress(40);
      try {
        const data = await func.fetchData(episodeNumber);
        setProgress(65);
        if (data.trimData) {
          setTitle(data.trimData.title);
          setContent(data.trimData.content);
          setURL(data.trimData.URL);
          setProgress(85);
        } else {
          console.error("No data found!");
          setURL(null);
        }
      } catch (error) {
        console.error("Error fetching episode:", error);
        setURL(null);
      }
      setProgress(100);
    };

    getData();
  }, [episodeNumber, setProgress]);

  const handleClick = () => {
    if (value >= 1 && value <= 806) {
      setEpisodeNumber(value);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === "Enter" &&
      event.target.value >= 1 &&
      event.target.value <= 806) {
      setEpisodeNumber(event.target.value);
    }
  };

  const handleChange = (event) => {
    setValue(event.target.value);
  };

  return (
    <div className="container my-4 d-flex flex-column gap-4 justify-content-center align-items-center text-center">

      <div className="d-flex gap-2 align-items-center">
        <input type="number"
          placeholder="Enter episode number"
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyPress}
          className="form-control"
          min="1"
          max="806"
          style={{ width: "14rem" }}
        />
        <button type="submit" className="btn btn-dark" onClick={handleClick}>
          Enter
        </button>
      </div>

      <div className="container d-flex flex-column justify-content-center align-items-center text-center gap-4">
        <img src="/image.webp" alt="chanakya-image" />
        <div>
          <h5>{title} - {content}</h5>
          <ReactPlayer
            url={URL}
            volume={0.5}
            playing
            controls
            playbackRate={playbackRate}
            height="50px"
          />
        </div>
      </div>

    </div>
  );
};

export default ChanakyaAudio;
