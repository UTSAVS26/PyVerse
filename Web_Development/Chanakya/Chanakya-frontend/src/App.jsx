import { useContext, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LoadingBar from "react-top-loading-bar";
import { Context } from "./context/Context";
import "./App.css";

import Navbar from "./components/shared/Navbar";
import Footer from "./components/shared/Footer";

import Home from "./pages/Home";
import About from "./pages/About";

import ChanakyaAudio from "./pages/resources/ChanakyaAudio";
import ChanakyaBook from "./pages/resources/ChanakyaBook";
import ChanakyaNews from "./pages/resources/ChanakyaNews";
import ChanakyaQuiz from "./pages/resources/ChanakyaQuiz";
import ChanakyaVideo from "./pages/resources/ChanakyaVideo";
import Login from "./pages/auth/Login";
import SignUp from "./pages/auth/SignUp";
import ChanakyaGpt from "./pages/resources/ChanakyaGpt";

import ForgotPassword from "./pages/auth/ForgotPassword";
import ResetPassword from "./pages/auth/ResetPassword";

function App() {
  const { progress, isDarkMode } = useContext(Context); // Assuming isDarkMode is provided in your context

  useEffect(() => {
    document.body.classList.toggle("dark", isDarkMode);
  }, [isDarkMode]);

  return (
    <div className={`d-flex flex-column ${isDarkMode ? "dark" : ""}`}>
      <Router>
        <Navbar />
        <LoadingBar height={3} color="#f11946" progress={progress} />

        <main className="container flex-grow-1 mt-4">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />

            <Route path="/resources/audio" element={<ChanakyaAudio />} />
            <Route path="/resources/book" element={<ChanakyaBook />} />
            <Route path="/resources/news" element={<ChanakyaNews />} />
            <Route path="/resources/quiz" element={<ChanakyaQuiz />} />
            <Route path="/resources/video" element={<ChanakyaVideo />} />
            <Route path="/resources/chanakyagpt" element={<ChanakyaGpt />} />
            {/* Authentication Pages */}
            <Route path="/auth/login" element={<Login />} />
            <Route path="/auth/signup" element={<SignUp />} />
            <Route path="/auth/forgot-password" element={<ForgotPassword />} />
            <Route path="/auth/resetpassword/:id/:token" element={<ResetPassword />} />
          </Routes>
        </main>

        <Footer />
      </Router>
    </div>
  );
}

export default App;
