import queue
import uuid
import zmq
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ..config import config
from .client import ModelClient
from ..env.chessboard import ChessBoard

# Create multiple clients and send prediction requests
client_queue = queue.Queue()
# TODO: uncomment this
CLIENTS=256
for _ in range(CLIENTS):
    client_queue.put(ModelClient())

def test(i):
    print(f"Testing query {i}")
    cb = ChessBoard()
    data = cb.get_plane()
    client = client_queue.get()
    policy, value = client.predict(
        0,
        data
    )
    print(f"query {i} received: policy={policy}, value={value}")
    client_queue.put(client)
    return policy, value
    
with ThreadPoolExecutor(max_workers=CLIENTS) as executor:
    futures = [executor.submit(test, i) for i in range(2000)]
    vals = [future.result()[1] for future in futures]