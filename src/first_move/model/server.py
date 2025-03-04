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
                    client_id = self.socket.recv()
                    _ = self.socket.recv()  # Empty delimiter
                    model_iter = int.from_bytes(self.socket.recv(4), 'little')
                    shape = np.frombuffer(self.socket.recv(), dtype=np.int32)
                    msg = self.socket.recv()
                    assert len(msg) == 1260
                    array = np.frombuffer(msg, dtype=np.int8).reshape(shape).astype(np.float32)
                    self.request_queue.put((client_id, model_iter, array))
                    i += 1
                    #print("Received:", i, self.request_queue.qsize())
                except Exception as e:
                    logger.error(f"Error receiving request: {e}")
                    traceback.print_exc()

            if self.response_socket in socks and socks[self.response_socket] == zmq.POLLIN:
                try:
                    client_id, packed_result = self.response_socket.recv_multipart(flags=zmq.NOBLOCK)
                    self.socket.send_multipart([client_id, b'', packed_result], flags=zmq.NOBLOCK)
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
                if True:
                    #batch_tensor = torch.from_numpy(batch_arrays[:current_batch_size]).to('cuda')
                    batch_tensor[:current_batch_size].copy_(torch.from_numpy(batch_arrays[:current_batch_size]))

                    with torch.no_grad():
                        policy_ary, value_ary = self.agent_models[model_iter](batch_tensor[:current_batch_size])
                    iteration += 1

                    policy_ary = policy_ary.cpu()
                    value_ary = value_ary.cpu()
                    # stream.synchronize()


                # Send results via PAIR socket
                for i in range(current_batch_size):
                    packed = self.pack_results(policy_ary[i], value_ary[i])
                    response_sender.send_multipart([client_ids[i], packed], flags=zmq.NOBLOCK)

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