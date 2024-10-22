import express from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { body, validationResult } from 'express-validator';

import User from '../models/User.js';
import fetchuser from '../middleware/fetchuser.js';
import nodemailer from 'nodemailer';



const JWT_SECRET = process.env.JWT_SECRET || 'Skanarul$123@';
const JWT_EXPIRY = process.env.JWT_EXPIRY || '1h';
const router = express.Router();
console.log("JWT_SECRET:", JWT_SECRET); 



// Route 1: Create a User using: POST "/api/auth/signup". No Login Required
router.post('/signup', [
  // Setting a validation so that input value will be verified before sending to the dataset
  body('name', 'Enter a valid name').isLength({ min: 3 }),
  body('email', 'Enter a valid email').isEmail(),
  body('password', 'Password must be atleast 8 characters').isLength({ min: 8 }),
], async (req, res) => {
  let success = false;
  // If there are errors, return Bad request and the errors
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ success, errors: errors.array() });
  }

  try {
    // Check whether the user with this email exists already
    let user = await User.findOne({ email: req.body.email });
    if (user) {
      return res.status(400).json({ success, error: "Sorry a user with this email already exists." });
    }

    // Implementing a hashing system to protect the password using 'bcrypt' package
    const salt = await bcrypt.genSalt(10);
    const securePassword = await bcrypt.hash(req.body.password, salt);

    // Create a new user
    user = await User.create({
      name: req.body.name,
      email: req.body.email,
      password: securePassword,
    });

    const data = {
      user: {
        id: user.id
      }
    }
   // Verify JWT_SECRET is correctly set
    console.log("JWT_SECRET:", JWT_SECRET);
    if (!JWT_SECRET) {
      throw new Error("JWT_SECRET is not defined");
    }
    // jwt will signature a token and give it to the user after login or say after verification/authentication
    const authToken = jwt.sign(data, JWT_SECRET,{ expiresIn: JWT_EXPIRY });
    success = true;
    res.json({ success, authToken });
  } catch (error) {
    console.error(error.message);
    res.status(500).send("Internal server error.");
  }
});

// Route 2: Authenticate a User using: POST "/api/auth/login". No Login Required
router.post('/login', [
  body('email', 'Enter a valid email').isEmail(),
  body('password', 'Password cannot be blank').exists(),
], async (req, res) => {
  let success = false;
  console.log('Login request received:', req.body);
  // If there are errors, return Bad request and the errors
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    console.log('Validation errors:', errors.array());
    return res.status(400).json({ success, errors: errors.array() });
  }

  const { email, password } = req.body;
  try {
    let user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ success, error: "Please try to login with correct credentials" });
    }
    const passwordCompare = await bcrypt.compare(password, user.password);
    if (!passwordCompare) {
      return res.status(400).json({ success, error: "Please try to login with correct credentials" });
    }
    const data = {
      user: {
        id: user.id
      }
    }
    const authtoken = jwt.sign(data, JWT_SECRET,{ expiresIn: JWT_EXPIRY });
    success = true;
    res.json({ success, authtoken });
  } catch (error) {
    console.error(error.message);
    res.status(500).send("Internal server error.");
  }
});

// Route 3: Get logged-in User Details using: POST "/api/auth/getuser". Login Required
router.post('/getuser', fetchuser, async (req, res) => {
  try {
    const userId = req.user.id;
    const user = await User.findById(userId).select("-password");
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    res.status(200).json(user);
  } catch (error) {
    console.error("Error fetching user details:",error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

//4 Route to send password reset link. use POST "/auth/restPasswordLink"
router.post('/resetPasswordLink', [
  body('email', 'Enter a valid email').isEmail(),
], async (req, res) => {
  try {
    const user = await User.findOne({ email: req.body.email });
    if (!user) {
      return res.status(400).json({ message: "User with this email does not exist" });
    }

    const secret = JWT_SECRET + user.password;
    const token = await jwt.sign({ email: user.email, id: user._id }, secret, { expiresIn: JWT_EXPIRY });

    const url = `${process.env.FRONTEND_URL}/auth/resetpassword/${user._id}/${token}`;
    const transporter = nodemailer.createTransport({
      host: 'smtp.gmail.com',
      port: 587,
      auth: {
        user: process.env.EMAIL,
        pass: process.env.PASSWORD
      }
    });

    await transporter.sendMail({
      to: user.email,
      subject: 'Password Reset Link',
      text: `Click on the link below to reset your password:\n${url}`,
      html: `<h2>Click on the link below to reset your password</h2><a href="${url}">Password Reset</a>`
    });

    res.status(200).json({ message: "Password reset link has been sent to your email" });
  } catch (error) {
    res.status(500).json({ message: "Something went wrong" });
  }
});

//5 Route to reset password. use POST "/api/auth/resetpassword/:id/:token"
router.post('/resetpassword/:id/:token', [
  body('password', 'Password must be at least 8 characters').isLength({ min: 8 }),
], async (req, res) => {
  const { id, token } = req.params;
  const { password } = req.body;

  try {
    const user = await User.findById(id);
    if (!user) {
      return res.status(400).json({ message: "User does not exist" });
    }

    const secret = JWT_SECRET + user.password;
    const decoded = jwt.verify(token, secret);

    if (decoded.id === user._id.toString()) {
      const salt = await bcrypt.genSalt(10);
      user.password = await bcrypt.hash(password, salt);
      await user.save();
      res.status(200).json({ message: "Password reset successful" });
    } else {
      res.status(400).json({ message: "Invalid token" });
    }
  } catch (error) {
    res.status(500).json({ message: "Something went wrong" });
  }
});

export default router;
