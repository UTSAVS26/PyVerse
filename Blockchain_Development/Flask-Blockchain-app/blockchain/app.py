from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from uuid import uuid4

from blockchain import Blockchain, MINING_REWARD, MINING_SENDER


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define blockchain Variables
blockchain = Blockchain()
node_identifier = str(uuid4()).replace("-", "")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/configure")
def configure():
    return render_template("configure.html")


@app.route("/transactions/new", methods=["POST"])
def new_transaction():
    values = request.form

    required = ["sender_address", "recipient_address", "amount", "signature"]
    if not all(k in values for k in required):
        return "Missing values", 400

    transaction_result = blockchain.submit_transaction(
        values["sender_address"],
        values["recipient_address"],
        values["amount"],
        values["signature"],
    )

    if not transaction_result:
        response = {"message": "Invalid Transaction!"}
        return jsonify(response), 406
    else:
        response = {
            "message": "Transaction will be added to Block " + str(transaction_result)
        }
        return jsonify(response), 201


@app.route("/transactions/get", methods=["GET"])
def get_transactions():
    transactions = blockchain.transactions

    response = {"transactions": transactions}
    return jsonify(response), 200


@app.route("/chain", methods=["GET"])
def full_chain():
    response = {
        "chain": blockchain.chain,
        "length": len(blockchain.chain),
    }
    return jsonify(response), 200


@app.route("/mine", methods=["GET"])
def mine():
    # We run the proof of work algorithm to get the next proof...
    last_block = blockchain.chain[-1]
    nonce = blockchain.proof_of_work()
    print(last_block)

    # We must receive a reward for finding the proof.
    blockchain.submit_transaction(
        sender_address=MINING_SENDER,
        recipient_address=blockchain.node_id,
        value=MINING_REWARD,
        signature="",
    )

    # Forge the new Block by adding it to the chain
    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(nonce, previous_hash)

    response = {
        "message": "New Block Forged",
        "block_number": block["index"],
        "transactions": block["transactions"],
        "nonce": block["nonce"],
        "previous_hash": block["previous_hash"],
    }
    return jsonify(response), 200


@app.route("/nodes/register", methods=["POST"])
def register_nodes():
    values = request.form
    nodes = values.get("nodes").replace(" ", "").split(",")

    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400

    for node in nodes:
        blockchain.register_node(node)

    response = {
        "message": "New nodes have been added",
        "total_nodes": [node for node in blockchain.nodes],
    }
    return jsonify(response), 201


@app.route("/nodes/resolve", methods=["GET"])
def consensus():
    replaced = blockchain.resolve_conflicts()

    if replaced:
        response = {"message": "Our chain was replaced", "new_chain": blockchain.chain}
    else:
        response = {"message": "Our chain is authoritative", "chain": blockchain.chain}
    return jsonify(response), 200


@app.route("/nodes/get", methods=["GET"])
def get_nodes():
    nodes = list(blockchain.nodes)
    response = {"nodes": nodes}
    return jsonify(response), 200
