import axios from 'axios';
const API_URL = process.env.API_URL;

export const registerUser = (userData) => {
    return axios.post(`${API_URL}/register`, userData);
};

export const loginUser = (credentials) => {
    return axios.post(`${API_URL}/token`, new URLSearchParams(credentials), {
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
    });
};

export const getAllUsers = (token) => {
    return axios.get(`${API_URL}/admin/users`, {
        headers: {
            Authorization: `Bearer ${token}`,
        },
    });
};

export const deleteUser = (userId, token) => {
    return axios.delete(`${API_URL}/admin/users/${userId}`, {
        headers: {
            Authorization: `Bearer ${token}`,
        },
    });
};
