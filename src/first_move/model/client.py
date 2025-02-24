import uuid
import zmq
import io
import sys
import numpy as np
from ..config import config
import os

class ModelClient:
    def __init__(self):
        self.context = zmq.Context()
        #print("Process ID:", os.getpid(), self.context)
        self.socket = self.context.socket(zmq.DEALER)  # DEALER socket for request/reply
        self.client_id = uuid.uuid4().bytes  # Unique client ID
        self.socket.setsockopt(zmq.IDENTITY, self.client_id)  # Set client ID
        self.socket.connect("ipc:///tmp/model_api.ipc")  # In-process communication

    def predict(self, model_iter, data):
        # Send data to the server
        self.socket.send_multipart([
            b'',
            model_iter.to_bytes(4, 'little'),
            np.array(data.shape, dtype=np.int32).tobytes(),
            data.tobytes()
        ])  # Serialize data and model_iter
        #self.socket.send_multipart([self.client_id, data.tobytes()])
        # Receive predictions from the server
        _, response = self.socket.recv_multipart()
        #response = self.socket.recv_multipart()
        #print(response)
        #arrays = np.load(io.BytesIO(response))
        #policy = arrays['p']
        #value = arrays['v']
        pv = np.frombuffer(response, dtype=np.float16)  # Deserialize combined data
        policy = pv[:-1]  # First 4 elements are policy
        value = pv[-1]   # Last element is the scalar value
        print(policy, value)
        #pv = np.frombuffer(response, dtype=np.float32)  # Deserialize policy and value
        #policy = pv[:4]  # Deserialize policy
        #value = pv[4:]  # Deserialize value
        return policy, value

    def close(self):
        self.socket.close()
        self.context.term()