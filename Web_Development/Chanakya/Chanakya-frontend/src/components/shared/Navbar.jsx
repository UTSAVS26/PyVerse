import { useContext, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { MdOutlineDarkMode, MdOutlineLightMode } from "react-icons/md";
import { Context } from "../../context/Context";
import "../shared/Navbar.css";

const Navbar = () => {
  const { isDarkMode, toggleTheme } = useContext(Context);
  const [isSticky, setIsSticky] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(true);

  const handleScroll = () => {
    setIsSticky(window.scrollY > 50);
  };

  useEffect(() => {
    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <nav
      className={`navbar navbar-expand-lg ${isSticky ? "fixed-top" : ""} ${
        isDarkMode ? "navbar-dark" : "navbar-light"
      }`}
      style={{
        backgroundColor: isDarkMode ? "rgba(0, 0, 0)" : "rgba(223, 223, 176)",
      }}
    >
      <div className="container-fluid">
        <div className="d-flex justify-content-between align-items-center w-100">
          <Link className="navbar-brand d-flex align-items-center" to="/">
            <img
              src="logo.webp"
              alt="Chanakya Image"
              className="me-2"
              style={{ width: "30px", height: "30px" }}
            />
            <span style={{ fontSize: "1.2rem", fontWeight: "bold" }}>
              चाणक्य नीति
            </span>
          </Link>

          <div className="d-flex align-items-center">
            <li
              className="nav-item nav-link"
              onClick={toggleTheme}
              style={{ cursor: "pointer", marginRight: "1rem" }}
            >
              {isDarkMode ? (
                <MdOutlineLightMode
                  style={{ fontSize: "1.5rem", color: "white" }}
                />
              ) : (
                <MdOutlineDarkMode
                  style={{ fontSize: "1.5rem", color: "black" }}
                />
              )}
            </li>

            <button
              className="navbar-toggler"
              type="button"
              onClick={toggleCollapse}
              aria-controls="navbarSupportedContent"
              aria-expanded={!isCollapsed}
              aria-label="Toggle navigation"
            >
              <span className="navbar-toggler-icon"></span>
            </button>
          </div>
        </div>

        <div
          className={`collapse navbar-collapse ${!isCollapsed ? "show" : ""}`}
          id="navbarSupportedContent"
        >
          <ul className="navbar-nav ms-auto mb-2 mb-lg-0">
            <li className="nav-item">
              <Link className="nav-link" to="/about">
                About
              </Link>
            </li>

            <li className="nav-item dropdown">
              <button
                className="nav-link dropdown-toggle btn btn-link"
                id="navbarDropdown"
                role="button"
                data-bs-toggle="dropdown"
                aria-expanded="false"
                onClick={() => console.log("Dropdown clicked")}
              >
                Resources
              </button>

              <ul className="dropdown-menu" aria-labelledby="navbarDropdown">
                <li>
                  <Link className="dropdown-item" to="/resources/chanakyagpt">
                    ChanakyaGpt
                  </Link>
                </li>
                <li>
                  <Link className="dropdown-item" to="/resources/audio">
                    Audios
                  </Link>
                </li>
                <li>
                  <Link className="dropdown-item" to="/resources/book">
                    Books
                  </Link>
                </li>
                <li>
                  <Link className="dropdown-item" to="/resources/news">
                    News
                  </Link>
                </li>
                <li>
                  <Link className="dropdown-item" to="/resources/quiz">
                    Quiz
                  </Link>
                </li>
                <li>
                  <Link className="dropdown-item" to="/resources/video">
                    Videos
                  </Link>
                </li>
              </ul>
            </li>

            <li className="nav-item">
              <Link className="nav-link" to="/auth/login">
                Login
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
