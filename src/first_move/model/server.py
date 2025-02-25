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
        for model_id in model_ids:
            self.agent_models[model_id] = ValuePolicyNet(config.model)  # CChessModel
            self.agent_models[model_id].load_model(model_path%(model_id))
            self.agent_models[model_id].eval()
        self.config = config.service
        self.done = False
        self.batch_size = self.config.batch_size_serve  # Batch size from config
        logger.info(f"Batch size: {self.batch_size}")
        self.batch_timeout = self.config.batch_timeout # Timeout for batching (in seconds)
        self.batch_lock = Lock()  # Lock for thread-safe batching
        self.batch_data = []  # Accumulated data for batching
        self.batch_clients = []  # Track which client sent which request
        self.client_sockets = defaultdict(list)  # Map client_id to their sockets
        self.address = "ipc:///tmp/model_api.ipc"  # Address for in-process communication
        #self.address = "ipc:///dev/shm/model_api.ipc"  # Address for in-process communication
        self.context = zmq.Context()
        self.request_queue = queue.Queue()  # Queue for incoming requests

    def start(self):
        # Router socket for handling multiple clients
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.SNDHWM, 5000)  
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
        i = 0
        while not self.done:
            try:
                # Receive client identity and request
                client_id = self.socket.recv()
                _ = self.socket.recv()  # Empty delimiter
                model_iter = int.from_bytes(self.socket.recv(4), 'little')  # Receive model iteration
                shape = np.frombuffer(self.socket.recv(), dtype=np.int32)  # Receive shape
                msg = self.socket.recv()  # Receive actual data
                assert(len(msg) == 1260)
                array = np.frombuffer(msg, dtype=np.int8).reshape(shape).astype(np.float32)

                # Queue the request with client ID
                self.request_queue.put((client_id, model_iter, array))
                i += 1
                print("Received:", i, self.request_queue.qsize())

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error receiving request: {e}")

    def batch_processor(self):
        # Wait for first request to determine shape
        client_id, model_iter, array = self.request_queue.get()
        expected_shape = array.shape
        #batch_tensor = torch.empty(self.batch_size, *expected_shape, device='cuda')
        
        # Handle first request
        batch_arrays = np.empty((self.batch_size, *expected_shape), dtype=np.float32)
        batch_arrays[0] = array  # Copy first array
        client_ids = [client_id]
        current_batch_size = 1

        try:
            # Collect remaining requests for first batch
            start_time = time.time()
            while current_batch_size < self.batch_size and \
                time.time() - start_time < self.batch_timeout:
                timeout_remaining = max(0.1, self.batch_timeout - (time.time() - start_time))
                # NOTE: here might deadlock if the last element is got and we try to get and blocked
                try:
                    client_id, model_iter, array = self.request_queue.get(timeout=min(1.0, timeout_remaining))
                except queue.Empty:
                    continue
                batch_arrays[current_batch_size] = array
                client_ids.append(client_id)
                current_batch_size += 1
            
            # Create sending thread
            send_queue = queue.Queue()
            send_thread = Thread(target=self.result_sender, args=(send_queue,))
            send_thread.daemon = True
            send_thread.start()
            tot = 0
            while not self.done:
                # Process current batch
                if current_batch_size > 0:
                    assert(current_batch_size <= self.batch_size)
                    with torch.cuda.stream(torch.cuda.Stream()):
                        start_time = time.time()
                        #for i, array in enumerate(batch_arrays):
                        batch_tensor = torch.from_numpy(batch_arrays[:current_batch_size]).to('cuda')#, non_blocking=True)
                        #batch_tensor = torch.from_numpy(np_arr).to('cuda')#, non_blocking=True)
                        mid_time = time.time()
                        with torch.no_grad():
                            policy_ary, value_ary = self.agent_models[model_iter](batch_tensor)
                        tot += len(client_ids)
                        end_time = time.time()
                        
                        # Queue results for sending
                        send_queue.put((client_ids, policy_ary.cpu(), value_ary.cpu()))
                        print("Processed:", time.time()-end_time, end_time-mid_time, mid_time-start_time, tot, len(client_ids), model_iter)
                        torch.cuda.empty_cache()
                else:
                    print("Empty batch")
                
                # Collect next batch
                client_ids = []
                current_batch_size = 0
                start_time = time.time()
                while current_batch_size < self.batch_size and time.time() - start_time < self.batch_timeout:
                    timeout_remaining = max(0.1, self.batch_timeout - (time.time() - start_time))
                    try:
                        # NOTE: if queue is empty, it will cause get to wait forever somehow
                        client_id, model_iter, array = self.request_queue.get(timeout=min(1.0, timeout_remaining))
                    except queue.Empty:
                        # No item received within the timeout, continue to check overall timeout
                        continue
                    batch_arrays[current_batch_size] = array  # Direct copy to pre-allocated array
                    client_ids.append(client_id)
                    current_batch_size += 1
        except:
            traceback.print_exc()

    def result_sender(self, send_queue):
        """Separate thread for sending results back to clients"""
        while not self.done:
            try:
                pack_start_time = time.time()
                client_ids, policy_ary, value_ary = send_queue.get()
                result_msgs = [self.pack_results(policy_ary[i], value_ary[i]) for i in range(len(client_ids))]
                pack_time = time.time() - pack_start_time
                
                send_start_time = time.time()
                for c, r in zip(client_ids, result_msgs):
                    self.socket.send_multipart([c, b'', r], flags=zmq.NOBLOCK)
                send_time = time.time() - send_start_time

                print("Pack time:", pack_time, "Send time:", send_time, "Items:", len(client_ids))
            except Exception as e:
                logger.error(f"Error sending results: {e}")

    @staticmethod
    def pack_results(policy, value):
        # Pack policies and values
        policy_data = policy.numpy().astype(np.float16).tobytes()
        value_data = value.numpy().astype(np.float16).tobytes()
        assert(len(policy_data) == 4172)
        assert(len(value_data) == 2)
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
            time.sleep(1)
    except KeyboardInterrupt:
        model_api.close()    
    model_api.wait()
