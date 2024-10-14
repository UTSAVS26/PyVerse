import React, { useState, useEffect } from 'react';
import { FaCircleUser } from "react-icons/fa6";
import toast from 'react-hot-toast';

import PrincipalService from '../../services/principal.service';
import AxiosApi from '../../services/axios.api';

import "./UserProfile.css";

const UserProfile = () => {
    const [user, setUser] = useState({
        id: '',
        name: '',
        email: '',
        role: '',
    });
    const [name, setName] = useState('');
    const [password, setPassword] = useState('');
    const [isEditing, setIsEditing] = useState(false);

    const fetchUserProfile = async () => {
        try {
            const userProfile = await PrincipalService.identity();
            setUser(userProfile);
            setName(userProfile.name);
        } catch (error) {
            console.error('Error Fetching User Details:', error);
            let errorMessage = 'Error Fetching User Details';
            if (error.response) {
                errorMessage = error.response.data.detail || 'Error Fetching User Details';
            } else if (error.request) {
                errorMessage = 'No response from server';
            } else {
                errorMessage = 'Error: ' + error.message;
            }
            toast.error(`Error Fetching User Details: ${errorMessage}`);
        }
    };

    useEffect(() => {
        fetchUserProfile();
    }, []);

    const handleEditProfile = () => {
        setIsEditing(true);
    };
    const handleSaveProfile = async (e) => {
        e.preventDefault();
        try {
            await AxiosApi.put('/users/me', {
                name,
                password,
            });
            toast.success("Profile Updated Successfully");
        } catch (error) {
            console.error('Error adding user:', error);
            let errorMessage = 'Registration Failed';
            if (error.response) {
                errorMessage = error.response.data.detail || 'Registration Failed';
            } else if (error.request) {
                errorMessage = 'No response from server';
            } else {
                errorMessage = 'Error: ' + error.message;
            }
            toast.error(`Registration Failed: ${errorMessage}`);
        }
        setIsEditing(false);
    }

    return (
        <div className='user-profile-container'>
            <h1>User Profile</h1>
            <div className="user-profile">
                <div className="profile-image-container">
                    <FaCircleUser size="100" />
                </div>
                <form className="profile-details">
                    <div className="form-group">
                        <label>ID:</label>
                        <input type="text" value={user.id} readOnly />
                    </div>
                    <div className="form-group">
                        <label>Name:</label>
                        <input
                            type="text"
                            value={name}
                            autoComplete='name'
                            onChange={(e) => setName(e.target.value)}
                            readOnly={!isEditing}
                        />
                    </div>
                    <div className="form-group">
                        <label>Email:</label>
                        <input type="email" value={user.email} readOnly />
                    </div>
                    <div className="form-group">
                        <label>Role:</label>
                        <input type="text" value={user.role} readOnly />
                    </div>
                    {isEditing && (
                        <div className="form-group">
                            <label>Password:</label>
                            <input
                                type="password"
                                value={password}
                                autoComplete='password'
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="Enter new password"
                            />
                        </div>
                    )}
                    <button
                        type="button"
                        onClick={isEditing ? handleSaveProfile : handleEditProfile}
                        className="edit-button"
                    >
                        {isEditing ? 'Save Profile' : 'Edit Profile'}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default UserProfile;
