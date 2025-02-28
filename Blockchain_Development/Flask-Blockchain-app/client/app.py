import binascii

import Crypto
import Crypto.Random
from Crypto.PublicKey import RSA
from flask import Flask, jsonify, request, render_template

from .transaction import Transaction


# Initialize Flask app
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/make/transaction")
def make_transaction():
    return render_template("make_transaction.html")


@app.route("/view/transactions")
def view_transaction():
    return render_template("view_transactions.html")


@app.route("/wallet/new", methods=["GET"])
def new_wallet():
    random_gen = Crypto.Random.new().read
    private_key = RSA.generate(1024, random_gen)
    public_key = private_key.publickey()
    response = {
        "private_key": binascii.hexlify(private_key.exportKey(format="DER")).decode(
            "ascii"
        ),
        "public_key": binascii.hexlify(public_key.exportKey(format="DER")).decode(
            "ascii"
        ),
    }

    return jsonify(response), 200


@app.route("/generate/transaction", methods=["POST"])
def generate_transaction():
    sender_address = request.form["sender_address"]
    sender_private_key = request.form["sender_private_key"]
    recipient_address = request.form["recipient_address"]
    value = request.form["amount"]

    transaction = Transaction(
        sender_address, sender_private_key, recipient_address, value
    )

    response = {
        "transaction": transaction.to_dict(),
        "signature": transaction.sign_transaction(),
    }

    return jsonify(response), 200
