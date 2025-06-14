import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./App.css";
import Header from "./components/Header";
import Registration from "./components/Registration";
import Login from "./components/Login";
import Dashboard from "./components/Dashboard";
import Home from "./components/Home";
import Courses from "./components/Courses";
import Coursepage from "./components/Coursepage";
import Footer from "./components/Footer";
import Roadmap from "./components/Roadmap";
import ProtectedRoute from "./components/ProtectedRoute"; // Import the ProtectedRoute component
import Scrolltotop from "./components/Scrolltotop";

function App() {
  const [user, setUser] = useState(undefined); // Change from null to undefined
  const [loading, setLoading] = useState(true); // Track loading state

  useEffect(() => {
    const savedUser = localStorage.getItem("user");
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
    setLoading(false); // Once user is set, stop loading
  }, []);

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem("user");
  };

  if (loading) {
    return <div>Loading...</div>; // Prevent rendering routes until user state is set
  }

  return (
    <Router>
      <Scrolltotop />
      <Header user={user} onLogout={handleLogout} />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route
            path="/Courses"
            element={
              <ProtectedRoute user={user}>
                <Courses />
              </ProtectedRoute>
            }
          />
          <Route
            path="/course/:id"
            element={
              <ProtectedRoute user={user}>
                <Coursepage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/Registration"
            element={<Registration setUser={setUser} />}
          />
          <Route
            path="/Roadmap"
            element={
              <ProtectedRoute user={user}>
                <Roadmap />
              </ProtectedRoute>
            }
          />
          <Route path="/Login" element={<Login setUser={setUser} />} />
          <Route
            path="/Dashboard"
            element={
              <ProtectedRoute user={user}>
                <Dashboard user={user} />
              </ProtectedRoute>
            }
          />
        </Routes>
      </main>
      <Footer />
    </Router>
  );
}

export default App;
