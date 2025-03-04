import os
import time
import threading
import multiprocessing as mp

from datetime import datetime, timedelta

import torch

from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm


class AsyncGradientSynchronizer:
    def __init__(
        self, 
        model: torch.nn.Module, 
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
        self.sync_lock = threading.Lock()  
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
        self.sync_thread = threading.Thread(target=self._sync_loop)
        self.sync_thread.daemon = True 
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
            return
            
        with torch.no_grad():
            with self.sync_lock:
                for buf, param in zip(self.gradient_buffer, self.model.parameters()):
                    if param.requires_grad and param.grad is not None:
                        buf.add_(param.grad)
    
    def synchronize_gradients(self):
        if self.world_size <= 1:
            return
        
        print(f"[Rank {self.rank}] Synchronizing gradients at {datetime.now().strftime('%H:%M:%S')}")
        
        for buf in self.gradient_buffer:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf.div_(self.world_size)
        
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
    if torch.cuda.is_available():
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("gloo")

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

def train_with_async_sync(model, train_loader, optimizer, epochs=10, sync_interval_seconds=60):
    
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
                
                if not sync.use_threading and sync.should_sync():
                    sync.synchronize_gradients()
                    sync.last_sync_time = time.time()
                
                optimizer.step()
                optimizer.zero_grad()
                
                if batch_idx % 10 == 0:
                    print(f'[Rank {os.environ.get("RANK", 0)}] Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}, Time: {datetime.now().strftime("%H:%M:%S")}')
                
                time.sleep(0.01)
    except Exception as e:
        print(f"Exception in training: {e}")
        raise
    finally:
        if not sync.sync_in_progress.is_set():
            sync.synchronize_gradients()
        
        sync.stop()

def main():
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
    
    train_with_async_sync(model, train_dataloader, optimizer, epochs=1000, sync_interval_seconds=5)
    
    print(f"Rank {rank} is done")
    cleanup()

if __name__ == "__main__":
    main()