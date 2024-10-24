import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';

const app = express();
const port = process.env.PORT || 3000;
const URI = process.env.MONGODB_URI;

// Middleware
const allowedOrigins = ['https://chanakya-web-application-ui-sk-anaruls-projects.vercel.app','https://chanakya-web-application-ui.vercel.app','http://localhost:5173'];

app.use(cors({
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
  credentials: true
}));
app.use(express.json());

// Database Connection
const connectDB = async () => {
  try {
    await mongoose.connect(URI);
    console.log("Chanakya DataBase Connected!");
  } catch (error) {
    console.error("Error connecting to the database", error);
    process.exit(1);
  }
};
connectDB();

// Available Routes
import auth from './routes/auth.js';
app.use('/api/auth', auth);
app.get('/', (req, res) => {
  res.send('Hello from Chanakya Niti Backend!');
});

// Start Server
app.listen(port, () => {
  console.log(`Chanakya Niti Backend listening on port http://localhost:${port}`);
});
