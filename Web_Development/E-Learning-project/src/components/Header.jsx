import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

function Header({ user, onLogout }) {
  const navigate = useNavigate();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const handleUserClick = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  const handleOptionClick = (path) => {
    navigate(path);
    setIsDropdownOpen(false); // Close dropdown after navigation
  };

  return (
    <header>
      <div className="navv">
        <nav>
          <div className="logo-image">
            <Link to="/">
              <img
                src="/Images/Logo.png"
                className="img-fluid"
                alt="Company Logo"
              />
            </Link>
          </div>
          <div className="ancors">
            <Link to="/">Home</Link>
            <Link to="/Courses">Courses</Link>
            <Link to="/Roadmap">Learn with AI</Link>
          </div>
          <div className="headd">
            {user ? (
              <div className="head-btn1">
                <div className="user-info" onClick={handleUserClick}>
                  <img
                    src="Images/User-icon.png"
                    alt="User Icon"
                    className="user-icon"
                  />
                  <h3 className="username">{user.firstname}</h3>
                </div>
                {isDropdownOpen && (
                  <div className="dropdown-menu">
                    <div
                      className="dropdown-item"
                      onClick={() => handleOptionClick("/")}
                    >
                      Home
                    </div>
                    <div
                      className="dropdown-item"
                      onClick={() => handleOptionClick("/Dashboard")}
                    >
                      Dashboard
                    </div>
                    <div
                      className="dropdown-item"
                      onClick={() => {
                        onLogout();
                        setIsDropdownOpen(false);
                        navigate("/Login");
                      }}
                    >
                      Logout
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="head-btn">
                <button className="nav-button">
                  <Link to="/Login" className="nav-link">
                    Login
                  </Link>
                </button>
              </div>
            )}
          </div>
        </nav>
      </div>
    </header>
  );
}

export default Header;
