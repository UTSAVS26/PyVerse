import React from 'react'

import './AdminDashboard.css'

const AdminDashboard = ({children}) => {
  return (
    <div className='admin-dashboard-container'>
      {children ? children : <h1 style={{ textAlign: "center", marginTop: "50px" }}>Welcome To AdminDashboard</h1>}
    </div>
  )
}

export default AdminDashboard
