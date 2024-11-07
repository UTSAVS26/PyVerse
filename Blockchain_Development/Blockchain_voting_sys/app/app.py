# app.py
from flask import Flask, render_template
from web3 import Web3

app = Flask(__name__)

# Connect to local Ethereum node or Infura
w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID'))

# Contract details
contract_address = 'YOUR_CONTRACT_ADDRESS'
contract_abi = '[/* YOUR_CONTRACT_ABI */]' # This will be [/* YOUR_CONTRACT_ABI */] like this.

# Create contract instance
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

@app.route('/')
def index():
    # Fetch candidates from the contract
    candidates = []
    candidates_count = contract.functions.candidatesCount().call()
    for i in range(1, candidates_count + 1):
        candidate = contract.functions.candidates(i).call()
        candidates.append({
            'id': candidate[0],
            'name': candidate[1],
            'voteCount': candidate[2]
        })
    return render_template('index.html', candidates=candidates)

if __name__ == '__main__':
    app.run(debug=True)
