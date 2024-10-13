import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import toast from 'react-hot-toast';

import AxiosApi from '../../services/axios.api';

import './RegistrationForm.css';

const RegistrationForm = () => {
    const navigate = useNavigate();
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [role, setRole] = useState('USER');

    const handleRegister = async (e) => {
        e.preventDefault();
        try {
            await AxiosApi.post('/register', {
                name,
                email,
                password,
                role: role,
            });
            toast.success("User Registered Successfully");
            navigate('/login');
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
    };


    return (
        <div className="registration-container">
            <form onSubmit={handleRegister} className="registration-form">
                <h2 className="registration-title">Register</h2>
                <label>Name:</label>
                <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                    autoComplete='name'
                    className="registration-input"
                />
                <label>Email:</label>
                <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    autoComplete='email'
                    className="registration-input"
                />
                <label>Password:</label>
                <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    autoComplete='password'
                    className="registration-input"
                />
                <label>Role:</label>
                <select
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="registration-select"
                >
                    <option value="USER">Normal User</option>
                    <option value="ADMIN">Admin</option>
                </select>
                <button type="submit" className="registration-button">Register</button>
                <p>Already Registered? <Link to="/login" className="login-link">Login Here</Link></p>
            </form>
        </div>
    );
};

export default RegistrationForm;
