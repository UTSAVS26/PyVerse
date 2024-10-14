import React from 'react'
import { Navigate } from 'react-router-dom';

const PrivateRoute = ({ role, children }) => {
    if (role) {
        return children;
    }
    else {
        return <Navigate to="/login" />
    }
}

export default PrivateRoute
