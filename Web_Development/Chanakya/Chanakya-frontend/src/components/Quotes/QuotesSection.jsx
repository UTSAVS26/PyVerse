import React, { useEffect, useState } from "react";
import quotes from "/src/database/quotes.json";
import "./QuoteSection.css";

const QuoteSection = () => {
  const [quote, setQuote] = useState("");
  const authInfo = process.env.QUOTES_API_KEY;

  useEffect(() => {
    fetchDailyQuote();
  }, []);

  const fetchDailyQuote = async () => {
    try {
      const response = await fetch("https://api.yourquoteapi.com/quotes?author=Chanakya", {
        headers: {
          "Authorization": authInfo,
        }
      });
      const data = await response.json();
      if (data && data.length > 0) {
        setQuote(data[0].quote);
      } else {
        setFallbackQuote();
      }
    } catch (error) {
      console.error("Error fetching the quote:", error);
      setFallbackQuote();
    }
  };

  const setFallbackQuote = () => {
    const randomIndex = Math.floor(Math.random() * quotes.length);
    setQuote(quotes[randomIndex].quote);
  };

  return (
    <div className="d-flex align-items-center justify-content-center">
      <div className="d-flex flex-column flex-md-row align-items-center">
        <div className="text-center text-md-start">
          <p className="quote-text">"{quote}"</p>
        </div>
      </div>
    </div>
  );
};

export default QuoteSection;
