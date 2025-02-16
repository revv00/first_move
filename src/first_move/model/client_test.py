import uuid
import zmq
import numpy as np
from config import config
from client import ModelClient

# Create multiple clients and send prediction requests
clients = []
for i in range(10):  # Simulate 10 clients
    client = ModelClient()
    data = np.random.rand(1, 10).astype(np.float32)  # Example input data
    policy, value = client.predict(data)
    print(f"Client {i} received: policy={policy}, value={value}")
    clients.append(client)

# Clean up
for client in clients:
    client.close()