<h1 align="center">CHANAKYA-NITI 📚</h1>

## Overview

CHANAKYA-NITI is a web application designed to provide an engaging platform for exploring and learning about the teachings of Chanakya, an ancient Indian philosopher, economist, and strategist.

<hr>

## Purpose and Motivation 🎯

This project aims to bring the timeless wisdom of Chanakya to a modern audience, making his teachings accessible and engaging through a digital platform.

<hr>

## Features ✨

- **Interactive Interface**: Engaging UI for exploring Chanakya's teachings.
- **Secure Backend**: Robust infrastructure to secure source code and multimedia content.
- **AI Integration**: Personalized recommendations and image processing.
- **User Authentication**: Secure and personalized user experiences.
- **Multimedia Content**: Audio files, books, videos on Chanakya’s life.
- **API Access**: Allows users to create their own Chanakya-Niti websites.
- **Language Translation**: AI model for translating content into multiple languages.

<hr>

## Frontend Technologies Used
- **React.js**: A JavaScript library for building user interfaces.
- **Redux**: State management for handling complex state across the application.
- **Axios**: Promise-based HTTP client for making API requests.
- **Material-UI**: React components for faster and easier web development.
- **React Router**: Declarative routing for React applications.
- **Vite**: A build tool that provides a faster and leaner development experience for modern web projects.

## Backend Technologies Used 
  - **Node.js**: Server-side JavaScript runtime.
  - **Express.js**: Web application framework for Node.js.
  - **MongoDB**: NoSQL database for storing application data.
  - **JWT**: JSON Web Tokens for secure user authentication.
  - **Groq Api**: AI model integration for personalized recommendations.

<hr>

 <details>
   <summary><h2>Rough Project Structure 👈</h2></summary>
Frontend:
D:.
│   App.css
│   App.jsx
│   index.css
│   main.jsx
│
├───components
│   ├───Home
│   │       VideoButton.jsx
│   │
│   ├───Quotes
│   │       QuoteSection.css
│   │       QuotesSection.jsx
│   │
│   └───shared
│           Card.jsx
│           Footer.jsx
│           Navbar.css
│           Navbar.jsx
│           Visitors.jsx
│
├───context
│       Context.jsx
│
├───css
│       Auth.css
│       ChanakyaNews.css
│       ChanakyaQuiz.css
│       Contributor.css
│       Footer.css
│
├───database
│       quotes.json
│
├───functions
│       RequestEpisode.module.js
│
└───pages
    │   About.jsx
    │   Home.jsx
    │
    ├───auth
    │       ForgotPassword.jsx
    │       Login.jsx
    │       ResetPassword.jsx
    │       SignUp.jsx
    │
    └───resources
            ChanakyaAudio.jsx
            ChanakyaBook.jsx
            ChanakyaGpt.jsx
            ChanakyaNews.jsx
            ChanakyaQuiz.jsx
            ChanakyaVideo.jsx
Backend:
├── Backend
│   ├── Express + Node + MongoDB
│   │   ├── User Authentication
│   │   │   └── JWT
│   │   │       └── Sign In/Sign Up
│   │   │           └── Database from MongoDB URL
│   │   ├── Database Integration
│   │   │   ├── MongoDB API
│   │   │   │   ├── Quotes resources
│   │   │   │   ├── Books resources
│   │   │   │   ├── Videos resources
│   │   │   │   ├── Audio resources
│   │   │   │   └── Contributors' records
│   │   │   └── GitHub API
│   │   └── API Access for logged-in users only
│   │       └── API hits capped at 1000
│   └── AI Integration
│   |   ├── Text-to-Speech Converter
│   |   │   └── For books resources
│   |   └── Language Translator
│   |       ├── For books
│   |       └── For audios
|   ├── Testing
|   └── Documentation
└── Database
    ├── Audio database
    ├── Videos database
    ├── Books database
    └── Quotes database
</details>

<hr>

## Installation Instructions 🛠️

1. Clone the repository:
   ```sh
   git clone https://github.com/<your-username>/chanakya-niti.git
   ```
2. Navigate to the project directory:
   ```sh
   cd chanakya-niti
   ```
3. Install dependencies:
   ```sh
   npm install
   ```
4. Start the development server:
   ```sh
   npm run dev(frontend) | npm start(backend) 
   ```## .env Structure:

## .env Structure:
### Frontend:
- `VITE_EPISODES_API_URL=https://api.github.com/repos/hack-boi/Chanakya/contents`
- `VITE_YOUTUBE_API_KEY=enter youtube api`
- `VITE_QUOTES_API_KEY=enter quotes api`
- `VITE_NEWS_API_KEY=enter news api`
- `VITE_BACKEND_URL=http://localhost:8081`

### Backend:
- `PORT=8081`
- `MONGODB_URI=mongodb://127.0.0.1:27017/database name`
- `FRONTEND_URL=http://localhost:5173`
- `JWT_EXPIRY=1h`
- `JWT_SECRET=Skanarul$123@`
- `PASSWORD=enter gmail auth pass key`
- `EMAIL=enter your gmail for forgot mail`
- `BACKEND_URL=http://localhost:8081/`


Thank you
