import sys
import traceback
import zmq
import logging
import threading
import queue
import time
import torch
import numpy as np
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from logging import getLogger
from collections import defaultdict
from .value_policy_net import ValuePolicyNet
from ..config import config

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

class CChessModelAPI:
    def __init__(self, config, model_path='./checkpoints/model_iter_%d_latest.pt', model_ids=[]):
        self.agent_models = {}
        self._model_path = model_path
        for model_id in model_ids:
            self.agent_models[model_id] = ValuePolicyNet(config.model)
            self.agent_models[model_id].load_model(model_path % model_id)
            self.agent_models[model_id].eval()
        self.config = config.service
        self.done = False
        self.batch_size = self.config.batch_size_serve
        logger.info(f"Batch size: {self.batch_size}")
        self.batch_timeout = self.config.batch_timeout
        self.batch_lock = Lock()
        self.batch_data = []
        self.batch_clients = []
        self.client_sockets = defaultdict(list)
        self.address = "ipc:///tmp/model_api.ipc"
        self.context = zmq.Context()
        self.request_queue = queue.Queue()
        self.response_socket = None  # PAIR socket for response communication

    def start(self):
        # Router socket for handling clients
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.SNDHWM, 5000)
        self.socket.bind(self.address)
        logger.info(f"Model server started on {self.address}")

        # PAIR socket for internal response communication
        self.response_socket = self.context.socket(zmq.PAIR)
        self.response_socket.bind("inproc://response_channel")

        # Start batch processing thread
        self.batch_thread = threading.Thread(target=self.batch_processor)
        self.batch_thread.daemon = True
        self.batch_thread.start()

        # Main loop for receiving requests
        self.receive_thread = threading.Thread(target=self.receive_requests)
        self.receive_thread.daemon = True
        self.receive_thread.start()

    def receive_requests(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        poller.register(self.response_socket, zmq.POLLIN)
        i = 0
        while not self.done:
            socks = dict(poller.poll(100))  # 100ms timeout to check self.done
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                try:
                    # Receive all parts in one call to reduce syscalls
                    parts = self.socket.recv_multipart()
                    client_id, _, model_iter_bytes, shape_bytes, msg = parts
                    
                    # Direct conversion to int without intermediate bytes object
                    model_iter = int.from_bytes(model_iter_bytes, 'little')
                    
                    # Use frombuffer without copying and reshape in one step
                    array = np.frombuffer(msg, dtype=np.int8)
                    array.shape = np.frombuffer(shape_bytes, dtype=np.int32)
                    
                    self.request_queue.put((client_id, model_iter, array.astype(np.float32)))
                    i += 1
                    #print("Received:", i, self.request_queue.qsize())
                except Exception as e:
                    logger.error(f"Error receiving request: {e}")
                    traceback.print_exc()

            if self.response_socket in socks and socks[self.response_socket] == zmq.POLLIN:
                try:
                    message = self.response_socket.recv_multipart(flags=zmq.NOBLOCK)
                    if message[0] == b'BATCH':
                        batch_size = np.frombuffer(message[1], dtype=np.int32)[0]
                        # Create views instead of copies
                        policy_data = np.frombuffer(message[2], dtype=np.float16).reshape(batch_size, -1)
                        value_data = np.frombuffer(message[3], dtype=np.float16).reshape(batch_size, -1)
                        client_id_length = len(message[4]) // batch_size
                        # Pre-allocate response buffer
                        response = policy_data[0].nbytes + value_data[0].nbytes
                        
                        # Use memoryview for faster slicing
                        client_ids_view = memoryview(message[4])
                        
                        # Avoid list comprehension and do direct indexing
                        for i in range(batch_size):
                            client_id = client_ids_view[i*client_id_length:(i+1)*client_id_length].tobytes()
                            # Combine policy and value data directly
                            self.socket.send_multipart([
                                client_id, 
                                b'',
                                message[2][i*policy_data[0].nbytes:(i+1)*policy_data[0].nbytes] + 
                                message[3][i*value_data[0].nbytes:(i+1)*value_data[0].nbytes]
                            ], flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue

    def batch_processor(self):
        # Create PAIR socket to send responses to receive thread
        response_sender = self.context.socket(zmq.PAIR)
        response_sender.connect("inproc://response_channel")
        try:
            # Initial request to determine shape
            client_id, model_iter, array = self.request_queue.get()
            expected_shape = array.shape
            batch_arrays = np.empty((self.batch_size, *expected_shape), dtype=np.float32)
            batch_tensor = torch.zeros((self.batch_size, *expected_shape), device='cuda')
            batch_arrays[0] = array
            client_ids = [client_id]
            current_batch_size = 1
            iteration = 0

            while not self.done:
                # Collect batch data
                start_time = time.time()
                while current_batch_size < self.batch_size and time.time() - start_time < self.batch_timeout:
                    try:
                        client_id, model_iter, array = self.request_queue.get(timeout=0.1)
                        if model_iter not in self.agent_models:
                            self.agent_models[model_iter] = ValuePolicyNet(config.model)
                            self.agent_models[model_iter].load_model(self._model_path % model_iter)
                            self.agent_models[model_iter].eval()
                        batch_arrays[current_batch_size] = array
                        client_ids.append(client_id)
                        current_batch_size += 1
                    except queue.Empty:
                        continue

                if current_batch_size == 0:
                    continue

                # Process batch
                # stream = torch.cuda.Stream()
                #with torch.cuda.stream(stream):
                #batch_tensor = torch.from_numpy(batch_arrays[:current_batch_size]).to('cuda')
                batch_tensor[:current_batch_size].copy_(torch.from_numpy(batch_arrays[:current_batch_size]))

                with torch.no_grad():
                    policy_ary, value_ary = self.agent_models[model_iter](batch_tensor[:current_batch_size])
                iteration += 1

                policy_ary = policy_ary.cpu()
                value_ary = value_ary.cpu()
                # stream.synchronize()


                # Send results via PAIR socket
                # Convert directly to float16 on GPU to reduce memory transfer
                policy_numpy = policy_ary[:current_batch_size].half().cpu().numpy()
                value_numpy = value_ary[:current_batch_size].half().cpu().numpy()
                
                # Pre-allocate the client_ids bytes
                client_ids_bytes = bytearray(sum(len(cid) for cid in client_ids))
                offset = 0
                for cid in client_ids:
                    client_ids_bytes[offset:offset + len(cid)] = cid
                    offset += len(cid)
                
                response_sender.send_multipart([
                    b'BATCH',
                    np.array(current_batch_size, dtype=np.int32).tobytes(),
                    policy_numpy.tobytes(),
                    value_numpy.tobytes(),
                    client_ids_bytes
                ], flags=zmq.NOBLOCK)

                # Reset for next batch
                batch_arrays = np.empty((self.batch_size, *expected_shape), dtype=np.float32)
                client_ids = []
                current_batch_size = 0

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            traceback.print_exc()
        finally:
            response_sender.close()

    @staticmethod
    def pack_results(policy, value):
        policy_data = policy.numpy().astype(np.float16).tobytes()
        value_data = value.numpy().astype(np.float16).tobytes()
        return policy_data + value_data

    def close(self):
        self.done = True
        self.socket.close()
        self.response_socket.close()
        self.context.term()

    def wait(self):
        self.receive_thread.join()
        self.batch_thread.join()

if __name__ == '__main__':
    model_ids = [int(i) for i in sys.argv[1].split(',')] if len(sys.argv) > 1 else []
    model_api = CChessModelAPI(config, model_ids=model_ids)
    model_api.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        model_api.close()
    model_api.wait()