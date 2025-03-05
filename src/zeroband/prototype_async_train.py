import argparse
import multiprocessing as mp
import os
import queue
import time
import threading

from contextlib import nullcontext
from datetime import datetime, timedelta

import torch

from torch import nn
from torch import distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, remote, rpc_async
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

class ParameterServer:
    def __init__(
        self,
        model
    ):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        self.lock = threading.Lock()
        self.update_count = 0
        self.last_report_time = time.time()

    def get_model_params(self):
        with self.lock:
            params = []
            for param in self.model.parameters():
                params.append(param.data.cpu().clone())
            return params

    def apply_gradients(self, gradients: list[torch.Tensor]):
        with self.lock:
            for param, grad in zip(self.model.parameters(), gradients):
                if param.grad is None:
                    param.grad = grad.to(param.device)
                else:
                    param.grad.add_(grad.to(param.device))

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.update_count += 1
            current_time = time.time()

            if current_time - self.last_report_time >= 5:
                print(f"[PS] Processed {self.update_count} gradient updates at {datetime.now().strftime('%H:%M:%S')}")
                self.last_report_time = current_time

            return self.update_count

class Worker:
    def __init__(
        self,
        model: nn.Module,
        ps_rref: RRef,
        rank: int,
        sync_interval_batches: int = 10,
    ):
        self.model = model
        self.ps_rref = ps_rref
        self.rank = rank
        self.sync_interval_batches = sync_interval_batches
        self.batches_processed = 0
        self.last_sync_batch = 0

        self.gradient_queue = queue.Queue()

        self.is_running = True
        self.push_thread = threading.Thread(target=self._push_gradients, daemon=True)
        self.push_thread.start()

        self.pull_thread = threading.Thread(target=self._pull_params_loop, daemon=True)
        self.pull_thread.start()

    def _push_gradients_loop(self):
        with self.is_running:
            try:
                try:
                    gradients = self.gradient_queue.get(timeout=0.1)

                    print(f"[Worker {self.rank}] Pushing gradients at {datetime.now().strftime('%H:%M:%S')}")
                    update_count = rpc_async(
                        self.ps_rref.owner(),
                        ParameterServer.apply_gradients,
                        args=(self.ps_rref, gradients)
                    )

                    self.gradient_queue.task_done()

                except queue.Empty:
                    pass
            except Exception as e:
                print(f"[Worker {self.rank}] Error in push loop: {e}")

            time.sleep(0.1)

    def _pull_params_loop(self):
        with self.is_running:
            try:
                if self.batches_processed - self.last_sync_batch >= self.sync_interval_batches:
                    print(f"[Worker {self.rank}] Requesting model params at {datetime.now().strftime('%H:%M:%S')}")

                    params = rpc_sync(
                        self.ps_rref.owner(), 
                        ParameterServer.get_model_params,
                        args=(self.ps_rref, )
                    ).wait()

                    with torch.no_grad():
                        for local_param, server_param in zip(self.model.parameters(), params):
                            local_param.data.copy_(server_param.to(local_param.device))

                    self.last_sync_batch = self.batches_processed
                    print(f"[Worker {self.rank}] Updated model params at {datetime.now().strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"[Worker {self.rank}] Error in pull loop: {e}")

            time.sleep(0.1)

    def process_batch(self, data: torch.Tensor, target: torch.Tensor):
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        gradients = []
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        gradients.append(param.grad.clone().cpu())
                    else:
                        gradients.append(torch.zeros_like(param.data).cpu())

        try:
            self.gradient_queue.put(gradients, block=False)
        except:
            print(f"[Worker {self.rank}] Gradient queue is full, skipping gradient push")

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        self.batches_processed += 1

        return loss.item()

    def stop(self):
        self.is_running = False
        if self.push_thread.is_alive():
            self.push_thread.join(timeout=5)
        if self.pull_thread.is_alive():
            self.pull_thread.join(timeout=5)

def setup_parameter_server(backend="gloo"):
    dist.init_process_group(backend)
    rpc.init_rpc(
        name=f"ps",
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            rpc_timeout=60
        )
    )

    print(f"Parameter server initialized using gloo backend")

def setup_worker():
    backend = "nccl"

    dist.init_process_group(backend)

    rpc.init_rpc(
        f"worker_{dist.get_rank()}",
        rank=rank,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            rpc_timeout=60
        )
    )

    print(f"Worker {dist.get_rank()} initialized using {backend} backend")


class QueueAsyncGradientSynchronizer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        sync_interval_seconds: int = 60,
        rank: int = None,
        world_size: int = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.sync_interval_seconds = sync_interval_seconds
        self.last_sync_time = time.time()
        self.rank = rank or (dist.get_rank() if dist.is_initialized() else 0)
        self.world_size = world_size or (dist.get_world_size() if dist.is_initialized() else 1)

        self.sync_queue = queue.Queue()
        self.is_running = False

        self._start_sync_thread()

    def _start_sync_thread(self):
        self.is_running = True
        # Have to use threading and not mp! mp doesn't work with CUDA 
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()

    def _sync_loop(self):
        while self.is_running:
            try:
                try:
                    gradients = self.sync_queue.get(timeout=0.1)
                    self._process_gradients(gradients)
                except queue.Empty:
                    # No task available, check if it's time for periodic sync
                    current_time = time.time()
                    if current_time - self.last_sync_time >= self.sync_interval_seconds:
                        self.request_sync()
                        self.last_sync_time = current_time
            except Exception as e:
                print(f"[Rank {self.rank}] Error in sync loop: {e}")

            time.sleep(0.1)

    def _process_gradients(self, gradients):
        try:
            print(f"[Rank {self.rank}] Processing gradients at {datetime.now().strftime('%H:%M:%S')}")

            if self.world_size > 1:
                for grad in gradients:
                    print("Before all reduce")
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                    print("After all reduce")
                    grad.div_(self.world_size)

            with torch.no_grad():
                param_idx = 0
                for param in self.model.parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            param.grad = gradients[param_idx].clone()
                        else:
                            param.grad.copy_(gradients[param_idx])
                        param_idx += 1

            print(f"[Rank {self.rank}] Complete synchronization at {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            print(f"[Rank {self.rank}] Error in processing gradients: {e}")

    def request_sync(self):
        gradients = []
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        gradients.append(param.grad.clone())
                    else:
                        gradients.append(torch.zeros_like(param.data))

        try:
            self.sync_queue.put(gradients, block=False)
        except queue.Full:
            print(f"[Rank {self.rank}] Sync queue is full, skipping sync request")

    def stop(self):
        self.is_running = False
        if self.sync_thread is not None and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)

# TODO: Failed
class AsyncGradientSynchronizer:
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim,
        sync_interval_seconds=60,
        use_threading=True,        
        rank=None,                 
        world_size=None            
    ):
        self.model = model
        self.optimizer = optimizer
        self.sync_interval_seconds = sync_interval_seconds
        self.use_threading = use_threading
        self.last_sync_time = time.time()
        self.gradient_buffer = None
        self.is_running = False
        self.sync_thread = None
        # self.sync_lock = threading.Lock()  
        self.sync_lock = nullcontext()
        self.sync_in_progress = threading.Event()
        self.rank = rank if rank is not None else dist.get_rank() if dist.is_initialized() else 0
        self.world_size = world_size if world_size is not None else dist.get_world_size() if dist.is_initialized() else 1
        
        self._init_gradient_buffer()
       
        if self.use_threading:
            self._start_sync_thread()
    
    def _init_gradient_buffer(self):
        self.gradient_buffer = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.gradient_buffer.append(torch.zeros_like(param.data))
    
    def _start_sync_thread(self):
        self.is_running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
    
    def _sync_loop(self):
        while self.is_running:
            current_time = time.time()
            if current_time - self.last_sync_time >= self.sync_interval_seconds:
                self.sync_in_progress.set() 
                with self.sync_lock:
                    try:
                        self.synchronize_gradients()
                        self.last_sync_time = current_time
                    except Exception as e:
                        print(f"[Rank {self.rank}] Error in sync loop: {e}")
                    finally:
                        # Add buffer timer here
                        time.sleep(5)
                self.sync_in_progress.clear()
                
            time.sleep(0.1)  
    
    def accumulate_gradients(self):
        if self.sync_in_progress.is_set():
            print(f"[Rank {self.rank}] Waiting for sync to complete at {datetime.now().strftime('%H:%M:%S')}")
            while self.sync_in_progress.is_set():
                time.sleep(0.1)
            print(f"[Rank {self.rank}] Sync completed, resuming at {datetime.now().strftime('%H:%M:%S')}")
            
        with torch.no_grad():
            with self.sync_lock:
                for buf, param in zip(self.gradient_buffer, self.model.parameters()):
                    if param.requires_grad and param.grad is not None:
                        buf.add_(param.grad)
    
    def synchronize_gradients(self):
        if self.world_size <= 1:
            return
        
        print(f"[Rank {self.rank}] Synchronizing gradients at {datetime.now().strftime('%H:%M:%S')}")
        

        print(f"[Rank {self.rank}] Before all reduce")
        for buf in self.gradient_buffer:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf.div_(self.world_size)
        print(f"[Rank {self.rank}] After all reduce")
        
        with torch.no_grad():
            for buf, param in zip(self.gradient_buffer, self.model.parameters()):
                if param.requires_grad:
                    if param.grad is None:
                        param.grad = buf.clone()
                    else:
                        param.grad.copy_(buf)
        
        for buf in self.gradient_buffer:
            buf.zero_()
    
    def should_sync(self):
        return time.time() - self.last_sync_time >= self.sync_interval_seconds
    
    def stop(self):
        if self.use_threading and self.is_running:
            self.is_running = False
            if self.sync_thread is not None and self.sync_thread.is_alive():
                self.sync_thread.join(timeout=5)


def setup():
    dist.init_process_group("cpu:gloo,cuda:nccl")

def cleanup():
    dist.destroy_process_group()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def create_dataloader(
    batch_size: int=32, 
    num_workers: int=1, 
    pin_memory: bool=True,
    shuffle: bool=True,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'shuffle': shuffle
    }

    dataset1 = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(dataset1, **transform_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **transform_kwargs)

    return train_loader, test_loader

def train_with_parameter_server(model):
    if rank == 0:
        device = "cpu"
        print(f"Parameter server is running on {device}")

        setup_parameter_server()

        model = Net().to(device)
        ps = ParameterServer(model)

        ps_rref = RRef(ps)

        rpc.shutdown()

    gpu


def train_with_queue_async(model, train_loader, optimizer, epochs=10, sync_interval_seconds=60):
    sync = QueueAsyncGradientSynchronizer(
        model, 
        optimizer, 
        sync_interval_seconds=sync_interval_seconds
    )
    device = next(model.parameters()).device
    
    try:
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                
                loss.backward()
                
                if batch_idx % 10 == 0:
                    sync.request_sync()
                
                optimizer.step()
                optimizer.zero_grad()
                
                if batch_idx % 10 == 0:
                    print(f'[Rank {os.environ.get("RANK", 0)}] Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}, Time: {datetime.now().strftime("%H:%M:%S")}')
                
                time.sleep(0.01)
    except Exception as e:
        print(f"Exception in training: {e}")
        raise
    finally:
        sync.stop()

def train_with_async_sync(model, train_loader, optimizer, epochs, sync_interval_seconds):
    
    sync = AsyncGradientSynchronizer(model, optimizer, sync_interval_seconds=sync_interval_seconds)
    device = next(model.parameters()).device
    
    try:
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                
                loss.backward()
                
                sync.accumulate_gradients()
                
                optimizer.step()
                optimizer.zero_grad()
                
                if batch_idx % 1 == 0:
                    print(f'[Rank {os.environ.get("RANK", 0)}] Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}, Time: {datetime.now().strftime("%H:%M:%S")}')
                
                time.sleep(0.01)
    except Exception as e:
        print(f"Exception in training: {e}")
        raise
    finally:
        if not sync.sync_in_progress.is_set():
            sync.synchronize_gradients()
        
        sync.stop()

def main(args):
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank} is running")
    print(f"World size: {world_size}")
    
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = Net().to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)
    
    train_dataloader, test_dataloader = create_dataloader()
    
    if args.async_method == "queue":
        train_with_queue_async(model, train_dataloader, optimizer, epochs=1000, sync_interval_seconds=10)
    else:
        train_with_async_sync(model, train_dataloader, optimizer, epochs=1000, sync_interval_seconds=10)
    
    print(f"Rank {rank} is done")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--async_method", type=str, default="queue", help="Method to use for async training")
    args = parser.parse_args()
    main(args)