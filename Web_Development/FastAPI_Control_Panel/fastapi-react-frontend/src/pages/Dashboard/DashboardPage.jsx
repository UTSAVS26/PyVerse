import React, { useEffect, useState,useCallback } from 'react'
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast'

//Services
import Auth from '../../services/auth'
import PrincipalService from '../../services/principal.service'
import PrincipalState from '../../services/principal.state'

//Components
import TopBar from '../../components/TopBar/TopBar'
import SideBar from '../../components/SideBar/SideBar'
import AdminDashboard from '../../components/AdminDashboard/AdminDashboard'
import UserDashboard from '../../components/UserDashboard/UserDashboard'
import UserTable from "../../components/UserTable/UserTable"
import UserProfile from '../../components/UserProfile/UserProfile'
import AddNewUser from '../../components/AddNewUser/AddNewUser'

//Styling Sheets
import './DashboardPage.css'

const DashboardPage = () => {
    const [role, setRole] = useState(sessionStorage.getItem("role"));
    const [selectedAction, setSelectedAction] = useState("Dashboard");
    const navigate = useNavigate();
    const authorize = useCallback (async() => {
        try {
            await Auth.authorize();
            await PrincipalService.isAuthenticated()
            const userIdentity = PrincipalState.getIdentity();
            setRole(userIdentity.role);
        } catch (error) {
            console.error('Error:', error);
            toast.error(`Error in Accesing Dashboard!`);
            navigate("/login");
        }

    },[navigate]);

    const AdminActions = [
        { "title": "Profile" },
        { "title": "Users" },
        { "title": "Add New User" },
        { "title": "Logout" },
    ];

    const UserActions = [
        { "title": "Profile" },
        { "title": "Logout" },
    ];
    const renderContent = () => {
        if (role === "ADMIN") {
            switch (selectedAction) {
                case "Profile":
                    return <UserProfile/>;
                case "Users":
                    return <UserTable/>;
                case "Add New User":
                    return <AddNewUser/>;
                case "Logout":
                    return;
                default:
                    return ;
            }
        } else {
            switch (selectedAction) {
                case "Profile":
                    return <UserProfile />;
                case "Logout":
                    return ;
                default:
                    return ;
            }
        }
    };
    const handleActionClick = (action) => {
        setSelectedAction(action);
    };
    const handleLogout = useCallback(async() => {
        toast.success("Logout Successfully!");
        Auth.cleanAuth();
        sessionStorage.removeItem("role");
        navigate("/login");
    },[navigate]);
    useEffect(() => {
        authorize();
    }, [authorize]);

    useEffect(() => {
        if (selectedAction === "Logout") {
            handleLogout();
        }
    }, [selectedAction, handleLogout]);
    return (
        <>
            <div className='dashboard-container'>
                <TopBar />
                <div className='content-container'>
                    <SideBar Actions={role === "ADMIN" ? (AdminActions) : (UserActions)} handleActionClick={handleActionClick}/>
                    {
                        role === "ADMIN" ? (
                            <AdminDashboard>{renderContent()}</AdminDashboard>
                        ) : (
                                <UserDashboard >{renderContent()}</UserDashboard>
                        )
                    }
                </div>
            </div>
        </>
    )
}

export default DashboardPage;
