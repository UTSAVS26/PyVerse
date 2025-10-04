import queue
import threading
import time
import pandas as pd
import json

# Simulate producer-consumer with queue (like Kafka topic)
txn_queue = queue.Queue()

def producer():
    """Simulate streaming transactions"""
    df = pd.read_csv('creditcard.csv').head(100)  # Sample data
    for _, txn in df.iterrows():
        txn_queue.put(json.dumps(txn.to_dict()))
        time.sleep(0.1)  # Simulate real-time
    txn_queue.put(None)  # End signal

def consumer(process_func):
    """Process incoming txns"""
    while True:
        msg = txn_queue.get()
        if msg is None:
            break
        txn = json.loads(msg)
        process_func(txn)  # e.g., score + explain

# Example process func (integrate with model later)
def dummy_process(txn):
    print(f"Processed txn: {txn['Amount']}")

# Run in threads
prod_thread = threading.Thread(target=producer)
cons_thread = threading.Thread(target=consumer, args=(dummy_process,))
prod_thread.start()
cons_thread.start()
prod_thread.join()
cons_thread.join()
print("Ingestion simulation complete.")
