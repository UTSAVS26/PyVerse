import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ref, get, child } from "firebase/database";
import { auth, database } from "./firebase"; // Import Firebase services

const Login = ({ setUser }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [showForgotPassword, setShowForgotPassword] = useState(false); // Toggle forgot password mode
  const navigate = useNavigate();

  // Handle login form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!email) {
      setError("Please enter your email address.");
      return;
    }

    // If in "forgot password" mode, log in with just email
    if (showForgotPassword) {
      try {
        const dbRef = ref(database); // Use the imported `database` service
        const snapshot = await get(child(dbRef, `Authentication`));
        if (snapshot.exists()) {
          const users = snapshot.val();
          const userKey = Object.keys(users).find(
            (key) => users[key].email === email
          );

          if (userKey) {
            const user = users[userKey];
            localStorage.setItem("user", JSON.stringify(user));
            setUser(user); // Update state
            setError("");
            navigate("/"); // Redirect to dashboard
          } else {
            setError("Email not found. Please register.");
          }
        } else {
          setError("No users found.");
        }
      } catch (error) {
        console.error("Error during login:", error);
        setError("An error occurred while logging in.");
      }
    } else {
      // Normal login with email and password
      if (!password) {
        setError("Please enter your password.");
        return;
      }

      try {
        const dbRef = ref(database); // Use the imported `database` service
        const snapshot = await get(child(dbRef, `Authentication`));
        if (snapshot.exists()) {
          const users = snapshot.val();
          const user = Object.values(users).find(
            (user) => user.email === email && user.password === password
          );

          if (user) {
            localStorage.setItem("user", JSON.stringify(user));
            setUser(user); // Update state
            setError("");
            navigate("/"); // Redirect to Homepage
          } else {
            setError("Invalid email or password.");
          }
        } else {
          setError("No users found.");
        }
      } catch (error) {
        console.error("Error during login:", error);
        setError("An error occurred while logging in.");
      }
    }
  };

  return (
    <div className="log-container">
      <div className="logform">
        <h2>Login Page</h2>
        {error && <div className="error-message">{error}</div>}

        {/* Login Form */}
        <form onSubmit={handleSubmit}>
          <label>Email:</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            required
          />

          {/* Show password field only if not in "forgot password" mode */}
          {!showForgotPassword && (
            <>
              <label>Password:</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
              />
            </>
          )}

          <button type="submit" className="log-btn">
            {showForgotPassword ? "Login with Email" : "Login"}
          </button>

          <div className="register-link">
            <span>Not Registered Yet?</span>
            <Link to="/Registration">Register</Link>
          </div>

          {/* Forgot Password Button */}
          <button
            type="button"
            className="forgot-password-btn"
            onClick={() => setShowForgotPassword(!showForgotPassword)}
          >
            <strong>
              {showForgotPassword ? "Back to Login" : "Forgot Password?"}
            </strong>
          </button>
        </form>
      </div>
    </div>
  );
};

export default Login;
