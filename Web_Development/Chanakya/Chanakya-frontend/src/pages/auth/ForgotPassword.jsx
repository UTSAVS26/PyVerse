import React, { useState } from "react";
import axios from "axios";

const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post(
        `${import.meta.env.VITE_BACKEND_URL}/api/auth/resetPasswordLink`,
        { email }
      );
      setMessage(response.data.message);
      setError("");
    } catch (error) {
      setError(error.response?.data?.message || "Something went wrong!");
      setMessage("");
    }
  };

  return (
    <div className="form-container">
      <p className="title">Forgot Password</p>
      <form className="form" onSubmit={handleSubmit}>
        {error && <p className="error">{error}</p>}
        {message && <p className="success">{message}</p>}
        <div className="input-group">
          <label htmlFor="email">Enter your email address</label>
          <input
            type="email"
            name="email"
            id="email"
            placeholder="Enter your email"
            autoComplete="off"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <button type="submit" className="sign mt-3">
          Send Reset Link
        </button>
      </form>
    </div>
  );
};

export default ForgotPassword;
