import React, { useState, useRef } from "react";
import { GoogleGenerativeAI } from "@google/generative-ai";
import ReactMarkdown from "react-markdown";
import jsPDF from "jspdf";

const Roadmap = () => {
  const [query, setQuery] = useState("");
  const [reply, setReply] = useState("");
  const [loading, setLoading] = useState(false);
  const [displayedReply, setDisplayedReply] = useState("");
  const [userLevel, setUserLevel] = useState("beginner");
  const [isDownloadEnabled, setIsDownloadEnabled] = useState(false);
  const responseContainerRef = useRef(null);

  // Initialize Generative AI API
  const genAI = new GoogleGenerativeAI(process.env.REACT_APP_GOOGLE_API_KEY);

  const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" }); // Use a supported model

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setIsDownloadEnabled(false);

    try {
      const prompt = `Create a learning roadmap for the course "${query}" tailored for a "${userLevel}" learner.`;
      console.log("Prompt:", prompt); // Debug: Log the prompt

      const result = await model.generateContent(prompt);
      console.log("API Response:", result); // Debug: Log the full API response

      const generatedReply = result.response.text();
      console.log("Generated Reply:", generatedReply); // Debug: Log the generated reply

      setReply(generatedReply);
      setQuery("");
      simulateTyping(generatedReply);
      setIsDownloadEnabled(true);
    } catch (error) {
      console.error("Error:", error);
      setReply("Sorry, something went wrong!");
    } finally {
      setLoading(false);
    }
  };

  const simulateTyping = (text) => {
    let index = 0;
    setDisplayedReply("");
    const interval = setInterval(() => {
      if (index < text.length) {
        setDisplayedReply((prev) => prev + text[index]);
        index++;
        autoscrollToBottom();
      } else {
        clearInterval(interval);
      }
    }, 4);
  };

  const autoscrollToBottom = () => {
    if (responseContainerRef.current) {
      window.scrollTo({
        top:
          responseContainerRef.current.offsetTop +
          responseContainerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  };

  const downloadAsPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(14);
    const formattedText = displayedReply
      .replace(/#/g, "")
      .replace(/\*\*/g, "")
      .replace(/\*/g, "•")
      .replace(/<br\s*\/?>/g, "\n");

    const textLines = doc.splitTextToSize(formattedText, 180);
    let y = 10;

    textLines.forEach((line, index) => {
      if (y > 270) {
        doc.addPage();
        y = 10;
      }
      doc.text(line, 10, y);
      y += 10;
    });

    doc.save("Roadmap.pdf");
  };

  return (
    <div className="chatbot-container">
      <h1 className="chatbot-header">Generate your personalized Roadmap</h1>
      <form onSubmit={handleSubmit} className="chatbot-form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., Python in 100 days"
          required
          className="chatbot-input"
        />
        <div className="chatbot-level">
          <label>
            <input
              type="radio"
              value="beginner"
              checked={userLevel === "beginner"}
              onChange={(e) => setUserLevel(e.target.value)}
            />
            Beginner
          </label>
          <label>
            <input
              type="radio"
              value="intermediate"
              checked={userLevel === "intermediate"}
              onChange={(e) => setUserLevel(e.target.value)}
            />
            Intermediate
          </label>
          <label>
            <input
              type="radio"
              value="advanced"
              checked={userLevel === "advanced"}
              onChange={(e) => setUserLevel(e.target.value)}
            />
            Advanced
          </label>
        </div>
        <button type="submit" className="chatbot-button">
          Submit
        </button>
      </form>
      <div
        ref={responseContainerRef}
        id="responseContent"
        className="chatbot-response"
      >
        <h2>Response:</h2>
        {loading ? (
          <p>
            We are crafting your Roadmap. It will take a moment, please wait...
          </p>
        ) : (
          <ReactMarkdown>
            {displayedReply.replace(/<br\s*\/?>/g, "\n")}
          </ReactMarkdown>
        )}
      </div>
      <div className="download-btn">
        <button
          onClick={downloadAsPDF}
          className="chatbot-button"
          disabled={!isDownloadEnabled}
        >
          Download as PDF
        </button>
      </div>
    </div>
  );
};

export default Roadmap;
