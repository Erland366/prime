import os
import time
import threading

from datetime import datetime, timedelta

import torch

from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

def setup():
    dist.init_process_group("cpu:gloo,cuda:nccl", init_method="env://")

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

class AsyncGradientSynchronizer:
    def __init__(
        self, 
        model, 
        sync_interval_seconds=60,  
        use_threading=True,        
        rank=None,                 
        world_size=None            
    ):
        self.model = model
        self.sync_interval_seconds = sync_interval_seconds
        self.use_threading = use_threading
        self.last_sync_time = time.time()
        self.gradient_buffer = None
        self.is_running = False
        self.sync_thread = None
        self.sync_lock = threading.Lock()  
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
                with self.sync_lock:  
                    self.synchronize_gradients()
                    self.last_sync_time = current_time
            time.sleep(0.1)  
    
    def accumulate_gradients(self):
        with torch.no_grad():
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
    
    def step(self, optimizer):
        sync_occurred = False
        
        
        if not self.use_threading and self.should_sync():
            with self.sync_lock:  
                self.synchronize_gradients()
                self.last_sync_time = time.time()
                sync_occurred = True
        
        
        optimizer.step()
        optimizer.zero_grad()
        
        return sync_occurred
    
    def stop(self):
        if self.use_threading and self.is_running:
            self.is_running = False
            if self.sync_thread is not None:
                self.sync_thread.join(timeout=5)


def create_dataloader(batch_size, input_size, dataset_length):
    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, size, length):
            self.len = length
            self.data = torch.randn(length, size)
            self.target = torch.randn(length, 5)

        def __getitem__(self, index):
            return self.data[index], self.target[index]

        def __len__(self):
            return self.len

    dataset = RandomDataset(input_size, dataset_length)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_with_async_sync(model, train_loader, optimizer, epochs=10, sync_interval_seconds=60):
    
    sync = AsyncGradientSynchronizer(model, sync_interval_seconds=sync_interval_seconds)
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
                
                
                sync_occurred = sync.step(optimizer)
                
                
                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
                
                
                time.sleep(0.01)
                
                
                if batch_idx > 0 and batch_idx % 100 == 0 and not sync.use_threading:
                    sync.synchronize_gradients()
                    sync.last_sync_time = time.time()
    except Exception as e:
        print(f"Exception in training: {e}")
        raise
    finally:
        
        sync.synchronize_gradients()
        
        sync.stop()


def main():
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank} is running")
    print(f"World size: {world_size}")
    
    model = ToyModel().to(f"cuda:{rank}")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    model = DDP(model, device_ids=[rank])
    
    dataloader = create_dataloader(batch_size=32, input_size=10, dataset_length=1000)
    
    train_with_async_sync(model, dataloader, optimizer, epochs=1000, sync_interval_seconds=5)
    
    print(f"Rank {rank} is done")
    cleanup()


if __name__ == "__main__":
    main()



















