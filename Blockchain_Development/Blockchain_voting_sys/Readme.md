# Blockchain Voting System

This project is a blockchain-based voting system that ensures transparency and immutability using smart contracts on the Ethereum blockchain. It includes a Solidity smart contract, a [Flask](https://flask.palletsprojects.com/en/2.3.x/) backend, and a JavaScript frontend for interacting with the contract.

## Features

- Secure voting using blockchain technology
- Dynamic candidate registration
- Owner-controlled voting period
- Real-time updates with event listening

## Prerequisites

- [Node.js](https://nodejs.org/en) and npm
- [Python](https://www.python.org/) and pip
- [MetaMask](https://metamask.io/) browser extension
- [Ganache](https://trufflesuite.com/ganache/) or access to an Ethereum test network

## Setup

### Smart Contract

1. **Install Truffle** (if not already installed):
   ```bash
   npm install -g truffle
2. **Compile and Deploy the Contract:**
- Navigate to the contracts directory.
- Run truffle compile to compile the Solidity contract.
- Run truffle migrate to deploy the contract to your local blockchain or test network.

## Backend (Flask)
1. **Install Flask and Web3.py:**
   ```bash
   pip install Flask web3
2.**Configure the Backend:**
- Update app.py with your deployed contract address and ABI.
- Ensure the Ethereum node URL is correctly set (e.g., Infura or local Ganache).
3. **Run the Flask Server:**
  ```bash
  python app.py
  ```
## Usage
1. **Open MetaMask** and connect to the same network where the contract is deployed.
2. **Access the Frontend:**
- Open your browser and navigate to http://localhost:8000 (or the port used by your HTTP server).
3. **Interact with the Voting System:**
- View candidates and their vote counts.
- Cast votes by entering a candidate ID.
- If you are the contract owner, toggle the voting status.
---

### Contributing
**Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.**

