import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ref, set, get } from "firebase/database";
import { database } from "./firebase"; // Import Firebase services

const Registration = ({ setUser }) => {
  const [data, setData] = useState({
    firstname: "",
    lastname: "",
    email: "",
    password: "",
    mobile: "",
    college: "",
    study: "",
  });

  const [error, setError] = useState({});
  const [success, setSuccess] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setData({ ...data, [e.target.name]: e.target.value });
    setError({ ...error, [e.target.name]: "" }); // Clear errors when user types
  };

  const validateForm = () => {
    let errors = {};
    if (!data.firstname) errors.firstname = "First name is required";
    if (!data.lastname) errors.lastname = "Last name is required";
    if (!data.email) errors.email = "Email is required";
    if (!data.password) errors.password = "Password is required";
    if (data.password.length < 6)
      errors.password = "Password must be at least 6 characters";
    if (!data.mobile) errors.mobile = "Mobile number is required";
    if (!/^\d{10}$/.test(data.mobile))
      errors.mobile = "Invalid mobile number. It should be 10 digits.";
    if (!data.college) errors.college = "College name is required";
    if (!data.study) errors.study = "Study field is required";

    setError(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSuccess("");

    // Validate the form
    if (!validateForm()) return;

    try {
      // Check if email already exists
      const emailRef = ref(
        database,
        `Authentication/${data.email.replace(/[.#$[\]]/g, "_")}`
      );
      const emailSnapshot = await get(emailRef);

      if (emailSnapshot.exists()) {
        setError({ email: "This email is already registered. Please login." });
        return;
      }

      // Check if mobile number already exists
      const mobileRef = ref(database, `AuthenticationByMobile/${data.mobile}`);
      const mobileSnapshot = await get(mobileRef);

      if (mobileSnapshot.exists()) {
        setError({
          mobile: "This mobile number is already registered. Please login.",
        });
        return;
      }

      // Save user data under email
      await set(emailRef, data);

      // Save user data under mobile number for uniqueness check
      await set(mobileRef, { email: data.email }); // Store only the email for reference

      setUser(data);
      setSuccess("Registration successful! Redirecting...");

      setTimeout(() => {
        navigate("/");
      }, 2000);
    } catch (error) {
      console.error("Registration Error:", error);
      setError({ form: "Failed to register. Please try again later." });
    }
  };

  return (
    <div className="reg-container">
      <div className="reg-form">
        <h2>Register</h2>
        {success && <div className="success-message">{success}</div>}

        <form onSubmit={handleSubmit}>
          <input
            type="text"
            name="firstname"
            value={data.firstname}
            onChange={handleChange}
            placeholder="First Name"
          />
          {error.firstname && (
            <div className="error-message">{error.firstname}</div>
          )}

          <input
            type="text"
            name="lastname"
            value={data.lastname}
            onChange={handleChange}
            placeholder="Last Name"
          />
          {error.lastname && (
            <div className="error-message">{error.lastname}</div>
          )}

          <input
            type="email"
            name="email"
            value={data.email}
            onChange={handleChange}
            placeholder="Email"
          />
          {error.email && <div className="error-message">{error.email}</div>}

          <input
            type="password"
            name="password"
            value={data.password}
            onChange={handleChange}
            placeholder="Password"
          />
          {error.password && (
            <div className="error-message">{error.password}</div>
          )}

          <input
            type="text"
            name="college"
            value={data.college}
            onChange={handleChange}
            placeholder="College"
          />
          {error.college && (
            <div className="error-message">{error.college}</div>
          )}

          <input
            type="text"
            name="study"
            value={data.study}
            onChange={handleChange}
            placeholder="Current Study"
          />
          {error.study && <div className="error-message">{error.study}</div>}

          <input
            type="text"
            name="mobile"
            value={data.mobile}
            onChange={handleChange}
            placeholder="Mobile Number"
          />
          {error.mobile && <div className="error-message">{error.mobile}</div>}

          <button type="submit">Register</button>
        </form>

        {error.form && <div className="error-message">{error.form}</div>}

        <div className="login-link">
          Already Registered? <a href="/Login">Login here</a>
        </div>
      </div>
    </div>
  );
};

export default Registration;
