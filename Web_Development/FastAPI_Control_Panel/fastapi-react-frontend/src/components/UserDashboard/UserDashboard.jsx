import React from 'react'

import './UserDashboard.css'

const UserDashboard = ({ children }) => {

  return (
    <div className='user-dashboard-container'>
      {children ? (children) : (<h1 style={{ textAlign: "center", marginTop: "50px" }}>Welcome To UserDashboard</h1>)}
    </div>
  )
}

export default UserDashboard;
