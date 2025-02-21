import sys
import zmq
import logging
import threading
import queue
import time
import torch
import numpy as np
from threading import Thread, Lock
from logging import getLogger
from collections import defaultdict
from .value_policy_net import ValuePolicyNet
from ..config import config

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

class CChessModelAPI:
    def __init__(self, config, model_path='./checkpoints/model_iter_%d_latest.pt', model_ids=[]):
        self.agent_models = {}
        for model_id in model_ids:
            self.agent_models[model_id] = ValuePolicyNet(config.model)  # CChessModel
            self.agent_models[model_id].load_model(model_path%(model_id))
        self.config = config.service
        self.done = False
        self.batch_size = self.config.batch_size_serve  # Batch size from config
        self.batch_timeout = self.config.batch_timeout # Timeout for batching (in seconds)
        self.batch_lock = Lock()  # Lock for thread-safe batching
        self.batch_data = []  # Accumulated data for batching
        self.batch_clients = []  # Track which client sent which request
        self.client_sockets = defaultdict(list)  # Map client_id to their sockets
        self.address = "ipc:///tmp/model_api.ipc"  # Address for in-process communication
        self.context = zmq.Context()
        self.request_queue = queue.Queue()  # Queue for incoming requests

    def start(self):
        # Router socket for handling multiple clients
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(self.address)
        logger.info(f"Model server started on {self.address}")

        # Start batch processing thread
        self.batch_thread = threading.Thread(target=self.batch_processor)
        self.batch_thread.daemon = True
        self.batch_thread.start()

        # Main loop for receiving requests
        self.receive_thread = threading.Thread(target=self.receive_requests)
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def receive_requests(self):
        while not self.done:
            try:
                # Receive client identity and request
                client_id = self.socket.recv()
                _ = self.socket.recv()  # Empty delimiter
                model_iter = int.from_bytes(self.socket.recv(4), 'little')  # Receive model iteration
                shape = np.frombuffer(self.socket.recv(), dtype=np.int32)  # Receive shape
                msg = self.socket.recv()  # Receive actual data
                # Unpack numpy array from bytes
                #array = np.frombuffer(msg[8:], dtype=np.float32)
                #shape = np.frombuffer(msg[:8], dtype=np.int32)
                #array = array.reshape(shape)
                array = np.frombuffer(msg, dtype=np.float32).reshape(shape)
                #print(f"Received request from {client_id} for model iteration {model_iter}:{array.shape}")

                # Queue the request with client ID
                self.request_queue.put((client_id, model_iter, array))

            except Exception as e:
                logger.error(f"Error receiving request: {e}")

    def batch_processor(self):
        while not self.done:
            batch_arrays = []
            client_ids = []
            start_time = time.time()

            # Collect requests until batch is full or timeout
            while len(batch_arrays) < self.batch_size and \
                time.time() - start_time < self.batch_timeout:
                try:
                    client_id, model_iter, array = self.request_queue.get(
                        timeout=self.batch_timeout - (time.time() - start_time)
                    )
                    batch_arrays.append(array)
                    client_ids.append(client_id)
                except queue.Empty:
                    break

            if not batch_arrays:
                continue

            try:
                # Process batch
                batch_data = np.stack(batch_arrays, axis=0)
                print("Stacked:", batch_data.shape)
                policy_ary, value_ary = self.agent_models[model_iter](torch.tensor(batch_data))
                print("Processed:", policy_ary.shape, value_ary.shape)

                # Send results back to clients
                start_idx = 0
                for i, client_id in enumerate(client_ids):
                    # Get client's results
                    client_policies = policy_ary[i]
                    client_values = value_ary[i]
                    
                    # Pack results
                    result_msg = self.pack_results(client_policies, client_values)

                    # Send results back to client
                    self.socket.send_multipart([
                        client_id,
                        b'',  # Empty delimiter
                        result_msg
                    ])
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Send error response to all clients in batch
                for client_id in client_ids:
                    self.socket.send_multipart([
                        client_id,
                        b'',
                        b'ERROR'
                    ])

    @staticmethod
    def pack_results(policy, value):
        # Pack policies and values
        policy_data = policy.detach().numpy().tobytes()
        value_data = value.detach().numpy().tobytes()

        return policy_data + value_data

    def close(self):
        self.done = True
        self.socket.close()
        self.context.term()
    
    def wait(self):
        self.receive_thread.join()
        self.batch_thread.join()

if __name__ == '__main__':
    # Start the model API
    model_ids = [int(i) for i in sys.argv[1].split(',')] if len(sys.argv) > 1 else []
    model_api = CChessModelAPI(config, model_ids=model_ids)
    model_api.start()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        model_api.close()    
    model_api.wait()
