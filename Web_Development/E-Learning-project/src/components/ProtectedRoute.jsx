import React, { useState, useEffect } from "react";
import { Navigate } from "react-router-dom";

const ProtectedRoute = ({ user, children }) => {
  const [showAlert, setShowAlert] = useState(false);
  const [redirect, setRedirect] = useState(false);

  useEffect(() => {
    if (!user) {
      setShowAlert(true);
      setTimeout(() => setRedirect(true), 2000); // Redirect after 2 seconds
    }
  }, [user]);

  if (!user) {
    return (
      <>
        {showAlert && (
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              background: "rgba(0, 0, 0, 0.5)",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              zIndex: 1000,
            }}
          >
            <div
              style={{
                background: "white",
                padding: "20px",
                borderRadius: "10px",
                boxShadow: "0px 0px 10px rgba(0, 0, 0, 0.3)",
                textAlign: "center",
                minWidth: "300px",
              }}
            >
              <p style={{ fontSize: "16px", marginBottom: "10px" }}>
                Please login to access this page.
              </p>
              <button
                onClick={() => setRedirect(true)}
                style={{
                  background: "#007bff",
                  color: "white",
                  border: "none",
                  padding: "10px 20px",
                  borderRadius: "5px",
                  cursor: "pointer",
                  fontSize: "14px",
                }}
              >
                OK
              </button>
            </div>
          </div>
        )}
        {redirect && <Navigate to="/" replace />}
      </>
    );
  }

  return children; // Render protected component only if user is authenticated
};

export default ProtectedRoute;
