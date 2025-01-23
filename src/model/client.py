import uuid
import zmq
import numpy as np
from config import config

class ModelClient:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)  # DEALER socket for request/reply
        self.client_id = uuid.uuid4().bytes  # Unique client ID
        self.socket.setsockopt(zmq.IDENTITY, self.client_id)  # Set client ID
        self.socket.connect("ipc:///tmp/model_api.ipc")  # In-process communication

    def predict(self, data):
        # Send data to the server
        self.socket.send_multipart([b'',data.tobytes()])  # Serialize data
        #self.socket.send_multipart([self.client_id, data.tobytes()])
        # Receive predictions from the server
        _, response = self.socket.recv_multipart()
        #response = self.socket.recv_multipart()
        pv = np.frombuffer(response, dtype=np.float32)  # Deserialize policy and value
        policy = pv[:4]  # Deserialize policy
        value = pv[4:]  # Deserialize value
        return policy, value

    def close(self):
        self.socket.close()
        self.context.term()