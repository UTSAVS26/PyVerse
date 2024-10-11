import React, { useState } from 'react';
import toast from 'react-hot-toast';

import AxiosApi from '../../services/axios.api';

import './AddNewUser.css';

const AddNewUser = () => {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [role, setRole] = useState('USER');

    const handleSubmit = async (event) => {
        event.preventDefault();
        try {
            await AxiosApi.post('/register', {
                name,
                email,
                password,
                role: role,
            });
            toast.success("User Added Successfully");
            setName('');
            setEmail('');
            setPassword('');
            setRole('USER');
        } catch (error) {
            console.error('Error adding user:', error);
            let errorMessage = 'Registration Failed';
            if(error.response){
                errorMessage = error.response.data.detail || 'Registration Failed';
            }else if(error.request){
                errorMessage='No response from server';
            }else {
                errorMessage = 'Error: ' + error.message;
            }
            toast.error(`Registration Failed: ${errorMessage}`);
        }
    };

    return (
        <div className="admin-actions-container">
            <h1>Add New User</h1>
            <form onSubmit={handleSubmit} className="admin-actions-form">
                <div className="form-group">
                    <label htmlFor="name">Name:</label>
                    <input
                        type="text"
                        id="name"
                        value={name}
                        autoComplete='name'
                        onChange={(e) => setName(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="email">Email (Username):</label>
                    <input
                        type="email"
                        id="email"
                        value={email}
                        autoComplete='email'
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="password">Password:</label>
                    <input
                        type="password"
                        id="password"
                        value={password}
                        autoComplete='password'
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label htmlFor="role">Role:</label>
                    <select
                        id="role"
                        value={role}
                        onChange={(e) => setRole(e.target.value)}
                    >
                        <option value="USER">User</option>
                        <option value="ADMIN">Admin</option>
                    </select>
                </div>
                <button type="submit" className="submit-button">Add User</button>
            </form>
        </div>
    );
};

export default AddNewUser;
