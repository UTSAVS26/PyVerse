from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

# Serve the HTML page for the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Serve the contract ABI and address
@app.route('/contract')
def contract_info():
    abi = '''[
        {
            "inputs": [],
            "stateMutability": "nonpayable",
            "type": "constructor"
        },
        {
            "inputs": [],
            "name": "Voting__AlreadyVoted",
            "type": "error"
        },
        {
            "inputs": [],
            "name": "Voting__IncorrectVoteIndex",
            "type": "error"
        },
        {
            "inputs": [],
            "name": "Voting__NotOwner",
            "type": "error"
        },
        {
            "inputs": [],
            "name": "addCandidate",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "getVotes",
            "outputs": [
                {
                    "internalType": "uint256[]",
                    "name": "",
                    "type": "uint256[]"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [
                {
                    "internalType": "uint256",
                    "name": "_voteIndex",
                    "type": "uint256"
                }
            ],
            "name": "vote",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "owner",
            "outputs": [
                {
                    "internalType": "address",
                    "name": "",
                    "type": "address"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [
                {
                    "internalType": "address",
                    "name": "",
                    "type": "address"
                }
            ],
            "name": "Voted",
            "outputs": [
                {
                    "internalType": "bool",
                    "name": "",
                    "type": "bool"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "candidates",
            "outputs": [
                {
                    "internalType": "uint256[]",
                    "name": "",
                    "type": "uint256[]"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]'''
    
    contract_address = "0x5FbDB2315678afecb367f032d93F642f64180aa3"  # Update this with your actual contract address
    
    return jsonify({
        "abi": abi,
        "address": contract_address
    })

if __name__ == '__main__':
    app.run(debug=True)
