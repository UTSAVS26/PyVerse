import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.transactions}{self.timestamp}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty):
        while not self.hash.startswith('0' * difficulty):
            self.nonce += 1
            self.hash = self.compute_hash()
        print(f"Block mined: {self.hash}")

class Blockchain:
    def __init__(self, difficulty=2):
        self.difficulty = difficulty
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, "0", "Genesis Block")
        self.chain.append(genesis_block)

    def add_block(self, transactions):
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, transactions)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != current_block.compute_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True

# Create a new blockchain and add some blocks to it
blockchain = Blockchain()
blockchain.add_block("Transaction 1: Alice pays Bob 5 BTC")
blockchain.add_block("Transaction 2: Bob pays Charlie 2 BTC")
blockchain.add_block("Transaction 3: Charlie pays Dave 1 BTC")

# Check the validity of the blockchain
print(f"Is blockchain valid? {blockchain.is_chain_valid()}")

# Display the blockchain with previous_hash and nonce included
for block in blockchain.chain:
    print(f"\nBlock {block.index}")
    print(f"Hash: {block.hash}")
    print(f"Previous Hash: {block.previous_hash}")
    print(f"Transactions: {block.transactions}")
    print(f"Nonce: {block.nonce}")
    print(f"Timestamp: {block.timestamp}")
