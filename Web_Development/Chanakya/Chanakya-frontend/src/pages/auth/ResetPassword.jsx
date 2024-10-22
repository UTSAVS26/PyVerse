import React, { useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";

const ResetPassword = () => {
  const { id, token } = useParams();
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post(
        `${
          import.meta.env.VITE_BACKEND_URL
        }/api/auth/resetpassword/${id}/${token}`,
        { password }
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
      <p className="title">Reset Password</p>
      <form className="form" onSubmit={handleSubmit}>
        {error && <p className="error">{error}</p>}
        {message && <p className="success">{message}</p>}
        <div className="input-group">
          <label htmlFor="password">New Password</label>
          <input
            type="password"
            name="password"
            id="password"
            placeholder="Enter new password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" className="sign mt-3">
          Save Password
        </button>
      </form>
    </div>
  );
};

export default ResetPassword;
