import React, { useState } from "react";

function Dashboard({ user }) {
  const randomNumber = Math.floor(10000 + Math.random() * 90000);
  const [isEditing, setIsEditing] = useState(false);
  const [userInfo, setUserInfo] = useState({
    firstname: user.firstname,
    lastname: user.lastname,
    email: user.email,
    mobile: user.mobile,
    college: user.college,
    study: user.study,
    location: "India",
    password: user.password,
  });

  const handleEdit = () => setIsEditing(!isEditing);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setUserInfo({ ...userInfo, [name]: value });
  };

  const handleSave = () => {
    setIsEditing(false);
    // Optionally, add logic to save the updated user details
  };

  return (
    <div className="dashboard-container">
      {/* Profile Section */}
      <div className="profile-card">
        <div className="profile-header">
          <img
            src="/Images/User-icon.png"
            alt="Profile"
            className="profile-pic"
          />
          <h1>
            {userInfo.firstname} {userInfo.lastname}
          </h1>
          <p>
            @{userInfo.firstname}
            {randomNumber}
          </p>
          <button className="edit-btn" onClick={handleEdit}>
            {isEditing ? "Cancel" : "Edit"}
          </button>
        </div>
      </div>

      {/* Personal Information Section */}
      <div className="info-card">
        <div className="info-header">
          <h2>Personal Information</h2>
        </div>
        <ul className="info-list">
          <li>
            <strong>Name:</strong>
            {isEditing ? (
              <>
                <input
                  type="text"
                  name="firstname"
                  value={userInfo.firstname}
                  onChange={handleInputChange}
                />
                <input
                  type="text"
                  name="lastname"
                  value={userInfo.lastname}
                  onChange={handleInputChange}
                />
              </>
            ) : (
              `${userInfo.firstname} ${userInfo.lastname}`
            )}
          </li>
          <li>
            <strong>Email:</strong>
            {isEditing ? (
              <input
                type="email"
                name="email"
                value={userInfo.email}
                onChange={handleInputChange}
              />
            ) : (
              userInfo.email
            )}
          </li>
          <li>
            <strong>Mobile No:</strong>
            {isEditing ? (
              <input
                type="text"
                name="mobile"
                value={userInfo.mobile}
                onChange={handleInputChange}
              />
            ) : (
              userInfo.mobile
            )}
          </li>
          <li>
            <strong>College:</strong>
            {isEditing ? (
              <input
                type="text"
                name="college"
                value={userInfo.college}
                onChange={handleInputChange}
              />
            ) : (
              userInfo.college
            )}
          </li>
          <li>
            <strong>Course:</strong>
            {isEditing ? (
              <input
                type="text"
                name="branch"
                value={userInfo.study}
                onChange={handleInputChange}
              />
            ) : (
              userInfo.study
            )}
          </li>
          <li>
            <strong>Location:</strong>
            {isEditing ? (
              <input
                type="text"
                name="location"
                value={userInfo.location}
                onChange={handleInputChange}
              />
            ) : (
              userInfo.location
            )}
          </li>
          <li>
            <strong>Password:</strong>
            {isEditing ? (
              <input
                type="text"
                name="password"
                value={userInfo.password}
                onChange={handleInputChange}
              />
            ) : (
              userInfo.password
            )}
          </li>
        </ul>
        {isEditing && (
          <button className="save-btn" onClick={handleSave}>
            Save
          </button>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
