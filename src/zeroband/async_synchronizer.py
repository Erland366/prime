import datetime
import threading
import multiprocessing as mp
import time

import torch

from torch import distributed as dist

class AsyncGradientSynchronizer:
    def __init__(
        self,
        model,
        sync_interval_seconds: int = 60,
        use_threading: bool = True,
        rank: int = None,
        world_size: int = None
    ):
        self.model = model
        self.sync_interval_seconds = sync_interval_seconds
        self.last_sync_time = time.time()
        self.gradient_buffer = None
        self.is_running = False
        self.sync_thread = None
        self.sync_lock = mp.Lock()
        self.rank = rank or (dist.get_rank() if dist.is_initialized() else 0)
        self.world_size = world_size or (dist.get_world_size() if dist.is_initialized() else 1)

        self._init_gradient_buffer()

    def _init_gradient_buffer(self):
        self.gradient_buffer = []
        for param in self.model.parameters():
            if param.requires_grad:
                self.gradient_buffer.append(torch.zeros_like(param.data))

    def _start_sync_thread(self):
        self.is_running = True
        self.sync_thread = mp.Process(target=self._sync_loop)
        self.sync_thread.start()

    def _sync_loop(self):
        while self.is_running:
            current_time = time.time()
            if current_time - self.last_sync_time >= self.sync_interval_seconds:
                with self.sync_lock:
                    self.synchronize_gradients()
                    self.last_sync_time = current_time
            time.sleep(1)

    def accumulate_gradients(self):
        with torch.no_grad():
            for buffer, param in zip(self.gradient_buffer, self.model.parameters()):
                if param.requires_grad and param.grad is not None:
                    buffer.add_(param.grad)

    def synchronize_gradients(self):
        if self.world_size <= 1:
            return

        print(f"Rank {self.rank} is synchronizing gradients at {datetime.now().strftime('%H:%M:%S')}")

        for buffer in self.gradient_buffer:
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            buf.div_(self.world_size)

        with torch.no_grad():
            for buffer, param in zip(self.gradient_buffer, self.model.parameters()):
                if param.requires_grad:
                    # TODO: Whether cloning is good
                    if param.grad is None:
                        param.grad = buffer.clone()
                    else:
                        param.grad.copy_(buffer)

        for buffer in self.gradient_buffer:
            buffer.zero_()

    def should_sync(self):
        return time.time() - self.last_sync_time >= self.sync_interval_seconds

    def step(self, optimizer):
        sync_occured = False

        if not self.use_threading and self.should_sync():
            with self.sync_lock:
                self.synchronize_gradients()
                self.last_sync_time = time.time()
                sync_occurred = True
