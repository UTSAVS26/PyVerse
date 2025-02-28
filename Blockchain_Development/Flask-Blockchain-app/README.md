# Blockchain implementation using Python.

This is a implementation of a basic blockchain structure in python, with all the description, and
documentation of it's working and things.

**NOTE**: It used to be a basic interaction API for finding the hashs, POW and the info. Currently
it's revamped into a full stack website with dummy payments mining and a better UI.

## Definition, and Representation of Blockchain

Here is the representation of a transaction in blockchain in Python.

```python
block = {
    'index': 1,
    'timestamp': 1506057125.900785,
    'transactions': [
        {
            'sender': "8527147fe1f5426f9dd545de4b27ee00",
            'recipient': "a77f5cdfa2934df3954a5c7c7da5df1f",
            'amount': 5,
        }
    ],
    'proof': 324984774000,
    'previous_hash': "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
}
```

Each new block contains within itself, the hash of the previous Block. This is crucial because it’s what
gives blockchains immutability. If an attacker corrupted an earlier Block in the chain then all subsequent
blocks will contain incorrect hashes.

### Creating new Blocks

When our Blockchain is instantiated we'll need to seed it with a genesis block, a block with
no predecessors. We’ll also need to add a "proof" to our genesis block which is the result of
mining (or proof of work).

### Understanding Proof of Work

A Proof of Work algorithm (PoW) is how new Blocks are created or mined on the blockchain.
The goal of PoW is to discover a number which solves a problem. The number must be difficult to
find but easy to verify computationally speaking by anyone on the network.
This is the core idea behind Proof of Work.

Let’s decide that the hash of some integer x multiplied by another y must end in 0.
So, `hash(x * y) = ac23dc...0` And for this simplified example, let’s fix `x = 5`.
Implementing this in Python:

```python
from hashlib import sha256

x = 5
y = 0  # Y needs to be calculated

while sha256(f'{x*y}'.encode()).hexdigest()[-1] != "0":
    y += 1

print(f'The solution is y = {y}')
```

The solution here is `y = 21`. Since, the produced hash ends in `0`:

```
hash(5 * 21) = 1253e9373e...5e3600155e860
```

In Bitcoin, the Proof of Work algorithm is called **Hashcash**. And it’s not too different from our
basic example above. It’s the algorithm that miners race to solve in order to create a new block.
In general, the difficulty is determined by the number of characters searched for in a string.
The miners are then rewarded for their solution by receiving a coin in a transaction.

The network is able to easily verify their solution.

This is what the request for a transaction will look like. It’s what the user sends to the server:

```json
{
  "sender": "my address",
  "recipient": "someone else's address",
  "amount": 5
}
```

## Tech Stack used

- `Flask` - A HTTP Gateway to expose our blockchain structure externally.
- `Requests` - A medium to check the HTTP Endpoint request and return JSON response.

## How to run the project?

- Install pipenv: `pip3 install pipenv`
- Make a env with Pipenv: `pipenv sync`
- Run the servers:
  - Run the miner server using `python -m frontend`
  - Run the clients using `python -m client <PORT-HERE>`

You can visit the site, play with the server, client and more, OR Use postman to Play and Mess with the
HTTP and JSON responses!

<div align="center">Made by Sunrit Jana with ❤️</div>
