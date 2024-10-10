import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';

import AxiosApi from '../../services/axios.api';

import './UserTable.css';

const UserTable = () => {
    const navigate = useNavigate();
    const [users, setUsers] = useState([]);

    const fetchUsers = useCallback(async() => {
        try {
            const token=sessionStorage.getItem('token')
            const data = await AxiosApi.get('/admin/users', {
                Authorization: `Bearer ${token}`,
            });
            setUsers(data);
            toast.success("Users Fetched Successfully!", { id: 'fetch-users-success' });
        } catch (error) {
            console.error('Error in fetching Data! :', error);
            let errorMessage = 'Error in fetching Data!';
            if (error.response) {
                errorMessage = error.response.data.detail || 'Error in fetching Data!';
            } else if (error.request) {
                errorMessage = 'No response from server!';
            } else {
                errorMessage = 'Error! : ' + error.message;
            }
            toast.error(`Registration Failed! : ${errorMessage}`, { id: 'fetch-users-error' });
            navigate('/login');
        }
    },[navigate]);

    const handleDelete = async (userId) => {
        const confirmDelete = window.confirm('Are you sure you want to delete this user?');
        if (confirmDelete) {
            try {
                const token = sessionStorage.getItem('token')
                await AxiosApi.del(`/admin/users/${userId}`, {
                    Authorization: `Bearer ${token}`,
                });
                setUsers(users.filter(user => user.id !== userId));
                toast.success('User deleted successfully');
            } catch (error) {
                console.error('Error in deleting the user! :', error);
                let errorMessage = 'Deletion Failed!';
                if (error.response) {
                    errorMessage = error.response.data.detail || 'Deletion Failed!';
                } else if (error.request) {
                    errorMessage = 'No response from server!';
                } else {
                    errorMessage = 'Error! : ' + error.message;
                }
                toast.error(`User deletion unsuccessful! ${errorMessage}`);
            }
        }
    };

    useEffect(() => {
        fetchUsers();
    }, [fetchUsers]);

    return (
        <>
            <div className="userTable-container">
                <h1>Users Table</h1>
                <table className="userTable">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Name</th>
                            <th>Email</th>
                            <th>Is Admin</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {users.map(user => (
                            <tr key={user.id}>
                                <td>{user.id}</td>
                                <td>{user.name}</td>
                                <td>{user.email}</td>
                                <td>{user.role}</td>
                                <td>
                                    {user.role !== 'ADMIN' && (
                                        <button
                                            className="deleteButton"
                                            onClick={() => handleDelete(user.id)}
                                        >
                                            Delete
                                        </button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </>
    );
};

export default UserTable;
