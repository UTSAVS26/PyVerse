pip install web3
#Importing required libraries
import json
import solcx
from web3 import Web3
from solcx import compile_standard, install_solc
import os
from dotenv import load_dotenv
from solcx import compile_standard
load_dotenv()
install_solc("0.6.0")
contract_source_code = """
pragma solidity ^0.6.0;
contract SimpleStorage {
 uint256 public storedData;
 bool private locked;
 modifier nonReentrant() {
 require(!locked, "ReentrancyGuard: reentrant call");
 locked = true;
 _;
 locked = false;
 }
 function set(uint256 x) public nonReentrant {
 storedData = x;
 }
 function get() public view returns (uint256) {
 return storedData;
 }
 function store(uint256 _value) public nonReentrant {
 storedData = _value;
 }
 function retrieve() public nonReentrant returns (uint256) {
 return storedData;
 }
}
"""
compiled_sol = compile_standard({
 "language": "Solidity",
 "sources": {"SimpleStorage.sol": {"content": contract_source_code}},
 "settings": {
 "outputSelection": {
 "*": {"*": ["abi", "evm.bytecode"]}
 }
 },
},
solc_version="0.6.0",
)
contract_abi = compiled_sol['contracts']['SimpleStorage.sol']['SimpleStorage']['abi']
contract_bytecode = compiled_sol['contracts']['SimpleStorage.sol']['SimpleStorage']['evm']['bytecode']['object']
confirmations=0
from web3 import Web3

# Connect to Ganache (replace 'http://127.0.0.1:7545' with your Ganache server URL)
w3 = Web3(Web3.HTTPProvider('HTTP://127.0.0.1:7545'))

# Get the chain ID
chain_id = w3.eth.chain_id

print(f"Chain ID: {chain_id}")
#USER ADDRESS
my_address = "0x96371042A15F4Eb0c495b495cfE2Bc0530eb645f" #This address and private key were taken from Ganache platform
#USER PRIVATE KEY
private_key = "0x4cf834d08df8cef9cad339fee979b7167d518035cd3536ffc0583b612cdcf546" #This private key was taken from Ganache platform
SimpleStorage = w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
nonce = w3.eth.get_transaction_count(my_address)
transaction_hash = SimpleStorage.constructor().build_transaction({
    "chainId": chain_id,
    "gasPrice": w3.eth.gas_price,
    "from": my_address,
    "nonce": nonce,
})
import time
def times():
    a = time.time()
    return a,time.ctime(a)
    #print("Timestamp created at :",a )
    #print("Time is (in GMT) : ",time.ctime(a))



#Contract creation
signed_tx = w3.eth.account.sign_transaction(transaction_hash, private_key=private_key)
tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print(f"Contract deployed to {tx_receipt.contractAddress}")
ts1,t1=times()
print("Timestamp created at :",ts1 )
print("Time is (in GMT) : ",t1)

simple_storage = w3.eth.contract(address=tx_receipt.contractAddress, abi=contract_abi)

from web3.middleware import geth_poa_middleware
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

sender_private_key = "0x4cf834d08df8cef9cad339fee979b7167d518035cd3536ffc0583b612cdcf546" #This address and private key were taken from Ganache platform
receiver_address = "0x6050B0fAA93c171E949F557098D81D5701558FC7" #This address and private key were taken from Ganache platform
sender_address = w3.eth.account.from_key(sender_private_key).address
w3.eth.default_account = sender_address

amount_to_transfer = int(input("Amount to transfer : "))
amount_in_wei = w3.to_wei(amount_to_transfer, 'ether')
transaction = {
    'to': receiver_address,
    'value': amount_in_wei,
    'gas': 21000,  # You may need to adjust the gas limit based on the actual gas cost
    'gasPrice': w3.to_wei('50', 'gwei'),  # Set the gas price according to network conditions
    'nonce': w3.eth.get_transaction_count(sender_address),
}
signed_transaction = w3.eth.account.sign_transaction(transaction, sender_private_key)
try:
    transaction_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)
    confirmations+=1
except:
    print("Insufficent funds")


if len(transaction_hash)==32:
    confirmations+=1

print(f"Transaction completed.......... \nTransaction hash: {transaction_hash.hex()}")
ts2,t2=times()
print("Timestamp created at :",ts2 )
print("Time is (in GMT) : ",t2)

from datetime import datetime
# Convert string timestamps to datetime objects
timestamp1 = datetime.utcfromtimestamp(ts1)
timestamp2 = datetime.utcfromtimestamp(ts2)

# Calculate the time difference
time_difference = timestamp2 - timestamp1
time_difference_Sec=time_difference.total_seconds()
print(f"Timestamp 1: {timestamp1}")
print(f"Timestamp 2: {timestamp2}")
print(f"Time difference: {time_difference}")
print("Time difference in seconds : ",time_difference_Sec)

if time_difference_Sec<=120:
    confirmations+=1

if confirmations==3:
    transaction_receipt = w3.eth.get_transaction_receipt(transaction_hash)
    print("Transaction receipt:\n")
    for key, value in transaction_receipt.items():
        print(f"            {key}: {value}")
        print()
else:
    print("Transaction is not proceeding due to unavailablitiy of fund or more time taken to process or invalid tranasaction hash")

x=0
if transaction_receipt is not None:
    if transaction_receipt["status"] == 1 :
        print("Transaction successful!")
        x+=1
    else:
        print("Transaction failed.")
else:
    print("Transaction not yet mined.")

# Get transaction details
if x==1:
    transaction = w3.eth.get_transaction(transaction_hash)
    print("Transaction details : ")
    for key, value in transaction.items():
        print(f"            {key}: {value}")
        print()

# Convert Wei to Ether
value_in_ether = w3.from_wei(transaction['value'], 'ether')
print(f"Value transferred in the transaction in ethers : {value_in_ether} Ether")
