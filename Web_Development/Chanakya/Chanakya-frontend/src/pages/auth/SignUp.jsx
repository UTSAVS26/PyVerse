import React, { useState } from 'react';
import axios from 'axios';
import '../../css/Auth.css';
import { FaGithub, FaGoogle, FaTwitter, FaEye, FaEyeSlash } from 'react-icons/fa6';

const URL = `${import.meta.env.VITE_BACKEND_URL}/api/auth/signup`;

const SignUp = () => {
	const [name, setName] = useState('');
	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');
	const [passwordVisible, setPasswordVisible] = useState(false);
	const [error, setError] = useState('');
	const [success, setSuccess] = useState('');

	const togglePasswordVisibility = () => {
		setPasswordVisible(!passwordVisible);
	};

	const handleSubmit = async (event) => {
		event.preventDefault();
		try {
			const response = await axios.post(URL, { name, email, password });
			if (response.data.success) {
				setSuccess('User created successfully!');
				window.location.href = '/auth/login';
			}
		} catch (error) {
			setError(error.response?.data?.error || 'Something went wrong!');
		}
	};

	return (
		<div className="form-container mb-4">
			<p className="title">Sign Up</p>

			<form className="form" onSubmit={handleSubmit}>
				{error && <p className="error">{error}</p>}
				{success && <p className="success">{success}</p>}

				<div className="input-group">
					<label htmlFor="name">Your Name</label>
					<input type="text" name="name" id="name" placeholder="Enter your name" autoComplete='off' value={name}
						onChange={(e) => setName(e.target.value)} required />
				</div>

				<div className="input-group">
					<label htmlFor="email">Email</label>
					<input type="email" name="email" id="email" placeholder="Enter your email" autoComplete='off' value={email}
						onChange={(e) => setEmail(e.target.value)} required />
				</div>

				<div className="input-group">
					<label htmlFor="password">Password</label>
					<div className="password-container">
						<input type={passwordVisible ? "text" : "password"} name="password" id="password" placeholder="Enter your password"
							value={password} onChange={(e) => setPassword(e.target.value)} required />
						<button type="button" className="password-toggle" onClick={togglePasswordVisibility}>
							{passwordVisible ? <FaEyeSlash /> : <FaEye />}
						</button>
					</div>
				</div>

				<button type="submit" className="sign mt-3">Sign Up</button>
			</form>

			<div className="social-message">
				<div className="line"></div>
				<p className="message">Sign up with social accounts</p>
				<div className="line"></div>
			</div>

			<div className="social-icons">
				<button aria-label="Sign up with Google" className="icon"><FaGoogle /></button>
				<button aria-label="Sign up with Twitter" className="icon"><FaTwitter /></button>
				<button aria-label="Sign up with GitHub" className="icon"><FaGithub /></button>
			</div>

		</div>
	)
}

export default SignUp;
