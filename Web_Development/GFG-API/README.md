# GFG Public API

GFG Public API is a Node.js application that fetches user data from GeeksforGeeks (GFG) using Puppeteer. This API allows you to retrieve details like user profile, rank, coding scores, solved problems, and more.
Deployment : https://gfg-api-tzvp.onrender.com/

## Features

- Fetch user profile data from GFG.
- Retrieve coding scores, ranks, and solved problems.
- Simple and easy-to-use API.

## Prerequisites

- Node.js (>= 14.20.1)
- npm or yarn

## Getting Started

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Shariq2003/GFG_API.git
    ```

2. Navigate to the project directory:

    ```sh
    cd GFG_API
    ```

3. Install the dependencies:

    ```sh
    npm install
    ```

### Running Locally

1. Install Puppeteer and its dependencies:

    ```sh
    npx puppeteer browsers install chrome
    ```

2. Start the server:

    ```sh
    npm start
    ```
    
    or
   
    ```sh
    npm run dev
    ```

4. Open your browser and navigate to `http://localhost:3000/<username>` to fetch the data for a specific GFG user.
