import React, { useState } from 'react';
import {Route, Routes } from 'react-router-dom';

//Pages
import HomePage from './pages/Home/HomePage';
import LoginPage from './pages/Login/LoginPage';
import DashboardPage from './pages/Dashboard/DashboardPage';

//Components
import PrivateRoute from './components/PrivateRoute';

//StylingnSheets
import './App.css'

const App = () => {
  const [role, setRole] = useState(sessionStorage.getItem("role"));
  function loginHandler(role){
    sessionStorage.setItem("role", role);
    setRole(role);
  }
  return (
    <>
      <div className='container'>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/register" element={<HomePage />} />
          <Route path="/login" element={<LoginPage loginHandler={loginHandler}/>} />
          <Route path="/dashboard" element={
            <PrivateRoute role={role}>
              <DashboardPage/>
            </PrivateRoute>
          } />
        <Route path="/*" element={<h3>Sorry this route is not defined yet</h3>} />
        </Routes>
      </div>
    </>
  );
};

export default App;