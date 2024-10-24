import React, { useState, useEffect } from "react";
import axios from "axios";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import "../../css/ChanakyaNews.css";
const ChanakyaNews = () => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchNews();
  }, []);

  const fetchNews = async () => {
    try {
      setLoading(true);
      const rss2jsonEndpoint = "https://api.rss2json.com/v1/api.json";
      const googleNewsRSS =
        "https://news.google.com/rss/search?q=archaryachanakya+OR+Arthashastra+OR+kautilya&hl=en-IN&gl=IN&ceid=IN:en";

      const response = await axios.get(rss2jsonEndpoint, {
        params: {
          rss_url: googleNewsRSS,
          api_key: import.meta.env.VITE_NEWS_API_KEY,
          count: 60,
        },
      });

      const newsItems = response.data.items.map((item) => ({
        title: item.title,
        link: item.link,
        pubDate: item.pubDate,
        description: item.description,
      }));

      const categorizedNews = categorizeNews(newsItems);
      setNews(categorizedNews);
      setLoading(false);
    } catch (err) {
      setError("Failed to fetch news. Please try again later.");
      setLoading(false);
    }
  };

  const categorizeNews = (newsItems) => {
    const categoryKeywords = {
      economics: ["economy", "finance", "wealth", "arthashastra", "economic"],
      politics: ["governance", "leadership", "strategy", "political"],
      history: ["history", "ancient", "mauryan", "empire", "historical"],
      teachings: ["wisdom", "teachings", "philosophy", "quotes"],
    };

    return newsItems
      .map((item) => {
        const content = (item.title + " " + item.description).toLowerCase();
        for (const [category, keywords] of Object.entries(categoryKeywords)) {
          if (keywords.some((keyword) => content.includes(keyword))) {
            return { ...item, category };
          }
        }
        return null; // Exclude items that don't match any category
      })
      .filter((item) => item !== null);
  };

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error-message">{error}</div>;

  const responsive = [
    {
      breakpoint: 1024,
      settings: {
        slidesToShow: 5,
      },
    },
    {
      breakpoint: 900,
      settings: {
        slidesToShow: 3,
      },
    },
    {
      breakpoint: 600,
      settings: {
        slidesToShow: 2,
      },
    },
    {
      breakpoint: 480,
      settings: {
        slidesToShow: 1,
      },
    },
  ];
  // Slider settings for the multi-item carousel
  const settings = {
    dots: false,
    infinite: true,
    speed: 500,
    slidesToShow: 4,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 2000,
    pauseOnHover: true,
    arrows: false,
    responsive,
  };
  return (
    <>
      <h1 className="mb-4 text-center">Latest News on Aacharya Chanakya</h1>
      <Slider {...settings}>
        {news.map((item, index) => (
          <div key={index} className="individualitem">
            <div className="nItem">
              <h5 className="card-des">{item.title}</h5>
              <div>
                <span className="badge bg-secondary mb-2 align-self-start ">
                  {item.category}
                </span>
                <p className="itemdate ">
                  Published: {new Date(item.pubDate).toLocaleDateString()}
                </p>
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-secondary mt-auto"
                >
                  Read more
                </a>
              </div>
            </div>
          </div>
        ))}
      </Slider>
    </>
  );
};

export default ChanakyaNews;
