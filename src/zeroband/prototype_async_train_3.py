import copy
import os
import random
import time
from collections import defaultdict, deque
import torch.multiprocessing as mp
from itertools import cycle

import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import tqdm

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@dataclass
class AsyncTrainerConfig:
    """Configuration for the AsyncTrainer classes"""
    # Model storage settings
    storage_mode: str = "cpu"  # 'cpu', 'gpu', 'disk', 'true_async'
    num_workers: int = 4
    staleness_max: int = 5
    alpha: float = 0.9  # Weight for global model update
    
    # Disk mode settings
    checkpoint_dir: str = "async_checkpoints"
    
    # Optimizer settings
    optimizer_class: Any = optim.SGD
    optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {"lr": 0.01, "momentum": 0.9})
    
    # Training settings
    total_steps: int = 1000
    batch_size: int = 32
    
    # Progress bar settings
    tqdm_update_freq: int = 1  # Update progress bar every N steps
    
    # Logging settings
    use_wandb: bool = True
    wandb_project: str = "async-training"
    wandb_name: Optional[str] = None
    wandb_log_freq: int = 10  # Log to wandb every N steps
    
    def __post_init__(self):
        # Validate settings
        valid_modes = ['cpu', 'gpu', 'disk', 'true_async']
        if self.storage_mode not in valid_modes:
            raise ValueError(f"storage_mode must be one of {valid_modes}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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
        return F.log_softmax(x, dim=1)


class AsyncTrainerBase:
    def __init__(
        self,
        model: nn.Module,
        config: AsyncTrainerConfig,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.num_workers = config.num_workers
        self.staleness_max = config.staleness_max
        self.alpha = config.alpha

        self.global_version = 0
        self.worker_versions = [0] * self.num_workers

        self.param_history = {}
        self.param_history_depth = self.staleness_max + 2

        for name, param in self.model.named_parameters():
            self.param_history[name] = deque(maxlen=self.param_history_depth)
            self.param_history[name].append((0, param.data.clone()))

        self.worker_active = [False] * self.num_workers
        self.worker_positions = [0] * self.num_workers
        self.loss_history = []
        self.worker_history = []
        
        self.worker_optimizers = [None] * self.num_workers
        
        # Initialize wandb if enabled
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            wandb_config = {
                "model_type": model.__class__.__name__,
                **vars(config)
            }
            wandb.init(project=config.wandb_project, name=config.wandb_name, config=wandb_config)

    def _snapshot_parameters(self):
        for name, param in self.model.named_parameters():
            self.param_history[name].append((self.global_version, param.data.clone()))

    def _get_parameter_version(self, version: int) -> dict:
        params_dict = {}
        for name, history in self.param_history.items():
            closest_version = 0
            closest_param = None

            for v, param in history:
                if v <= version and v >= closest_version:
                    closest_version = v
                    closest_param = param

            if closest_param is not None:
                params_dict[name] = closest_param

        return params_dict

    def initialize_worker_models(self) -> None:
        raise NotImplementedError

    def get_worker_model(self, worker_id: int) -> nn.Module:
        raise NotImplementedError

    def update_worker_model(self, model_or_state: nn.Module, worker_id: int) -> None:
        raise NotImplementedError

    def update_global_model(self, worker_model: nn.Module, worker_id: int) -> None:
        with torch.no_grad():
            for g_param, w_param in zip(self.model.parameters(), worker_model.parameters()):
                if g_param.device != w_param.device:
                    w_param = w_param.to(g_param.device)
                g_param.data = self.alpha * g_param.data + (1 - self.alpha) * w_param.data

        self.global_version += 1
        self._snapshot_parameters()
        
        # Update worker version to latest AFTER contributing
        self.worker_versions[worker_id] = self.global_version
        
        self.worker_history.append((self.global_version, worker_id))
        
        tqdm.write(f"Global model updated to version {self.global_version} by worker {worker_id}")
    
    def train_step(self, dataloader, total_steps=None, optimizer_class=None, 
                  optimizer_kwargs=None):
        if total_steps is None:
            total_steps = self.config.total_steps
        if optimizer_class is None:
            optimizer_class = self.config.optimizer_class
        if optimizer_kwargs is None:
            optimizer_kwargs = self.config.optimizer_kwargs
            
        step = 0
        self.initialize_worker_models()
        
        worker_dataloaders = [cycle(dataloader) for _ in range(self.num_workers)]
        
        main_pbar = tqdm(total=total_steps, desc="Overall Progress", position=0)
        
        worker_pbars = {}
        for worker_id in range(self.num_workers):
            pbar = tqdm(
                total=0,  
                desc=f"Worker {worker_id}",
                position=worker_id + 1
            )
            pbar.set_postfix(version=self.worker_versions[worker_id], loss=0.0, active="No")
            worker_pbars[worker_id] = pbar
        
        global_pbar = tqdm(total=total_steps, desc="Global Model", position=self.num_workers + 1)
        global_pbar.n = self.global_version
        global_pbar.refresh()
    
        try:
            while step < total_steps:
                # Select a worker randomly
                worker_id = random.randint(0, self.num_workers - 1)
                
                # Check if worker needs initialization
                if self.worker_optimizers[worker_id] is None:
                    worker_model = self.get_worker_model(worker_id)
                    self.worker_optimizers[worker_id] = optimizer_class(
                        worker_model.parameters(), **optimizer_kwargs)
                else:
                    worker_model = self.get_worker_model(worker_id)
                worker_device = next(worker_model.parameters()).device
                
                if not self.worker_active[worker_id]:
                    self.worker_active[worker_id] = True
                    
                    staleness = random.randint(0, self.staleness_max)
                    target_version = max(0, self.global_version - staleness)
    
                    self.worker_versions[worker_id] = target_version
                    worker_pbars[worker_id].set_postfix({
                        "version": target_version, 
                        "loss": 0.0,
                        "active": "Yes"
                    })
                    worker_pbars[worker_id].refresh()
    
                data, target = next(worker_dataloaders[worker_id])
                data, target = data.to(self.device), target.to(self.device)
                self.worker_positions[worker_id] += 1
                
                # Only update tqdm progress at configured frequency
                if self.worker_positions[worker_id] % self.config.tqdm_update_freq == 0:
                    worker_pbars[worker_id].total = self.worker_positions[worker_id]
                    worker_pbars[worker_id].n = self.worker_positions[worker_id] - 1
                    worker_pbars[worker_id].refresh()
                
                # Get optimizer for this worker
                # TODO: THIS IS STILL RECREATE THE OPTIMIZER EVERY TIME
                # WHICH MEANS WE LOSE THE STATE OF THE OPTIMIZER
                optimizer = optimizer_class(worker_model.parameters(), **optimizer_kwargs)
                self.worker_optimizers[worker_id] = optimizer

                optimizer.zero_grad()
                loss = self._compute_loss(worker_model, data, target)
                loss.backward()
                optimizer.step()
                self.loss_history.append((step, worker_id, loss.item()))
                
                current_postfix = worker_pbars[worker_id].postfix if isinstance(worker_pbars[worker_id].postfix, dict) else {}
                current_postfix["loss"] = f"{loss.item():.4f}"
                worker_pbars[worker_id].set_postfix(current_postfix)
                
                prev_global_version = self.global_version
                
                self.update_worker_model(worker_model, worker_id)
                
                if self.global_version > prev_global_version:
                    global_pbar.n = self.global_version
                    global_pbar.refresh()
                
                # Log metrics to wandb
                if self.use_wandb and (step % self.config.wandb_log_freq == 0 or step == total_steps - 1):
                    metrics = {
                        "step": step,
                        "global_version": self.global_version,
                        "worker_loss": loss.item(),
                        "worker_id": worker_id,
                        "worker_version": self.worker_versions[worker_id],
                        "staleness": self.global_version - self.worker_versions[worker_id],
                    }
                    
                    # Add worker-specific metrics
                    for i in range(self.num_workers):
                        metrics[f"worker_{i}_active"] = self.worker_active[i]
                        metrics[f"worker_{i}_version"] = self.worker_versions[i]
                        metrics[f"worker_{i}_staleness"] = self.global_version - self.worker_versions[i]
                    
                    # Log number of active workers
                    metrics["active_workers"] = sum(self.worker_active)
                    wandb.log(metrics)
                
                if random.random() < 0.2:  # 20% chance of worker finishing its task
                    self.worker_active[worker_id] = False
                    worker_pbars[worker_id].set_postfix({
                        "version": self.worker_versions[worker_id], 
                        "loss": f"{loss.item():.4f}",
                        "active": "No"
                    })
                
                step += 1
                if step % self.config.tqdm_update_freq == 0:
                    main_pbar.update(self.config.tqdm_update_freq)
                
        finally:
            main_pbar.close()
            for pbar in worker_pbars.values():
                pbar.close()
            global_pbar.close()
            
            for optimizer in self.worker_optimizers:
                if optimizer is not None:
                    del optimizer
            self.worker_optimizers = [None] * self.num_workers
    
        return self.loss_history

    def _compute_loss(self, model, data, target):
        model.train()
        output = model(data)
        if hasattr(model, "loss_fn"):
            loss = model.loss_fn(output, target)
        else:
            loss = F.cross_entropy(output, target)
        
        return loss

    def evaluate(self, test_loader):
        """Evaluate the global model on test data"""
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        
        # Log evaluation metrics to wandb
        if self.use_wandb:
            wandb.log({
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "global_version": self.global_version
            })
        
        return test_loss, accuracy
    
    def cleanup(self):
        for name in self.param_history:
            self.param_history[name].clear()
        
        for i in range(len(self.worker_optimizers)):
            if self.worker_optimizers[i] is not None:
                del self.worker_optimizers[i]
                self.worker_optimizers[i] = None


class GPUAsyncTrainer(AsyncTrainerBase):
    def __init__(
        self,
        model: nn.Module,
        config: AsyncTrainerConfig,
    ):
        super().__init__(model, config)
        self.worker_models = None

    def initialize_worker_models(self) -> None:
        self.worker_models = [copy.deepcopy(self.model) for _ in range(self.num_workers)]
        print(f"Initialized {self.num_workers} worker models in GPU memory")

    def get_worker_model(self, worker_id: int) -> nn.Module:
        version_params = self._get_parameter_version(self.worker_versions[worker_id])

        with torch.no_grad():
            for name, param in self.worker_models[worker_id].named_parameters():
                if name in version_params:
                    param.data.copy_(version_params[name].to(self.device))

        return self.worker_models[worker_id]

    def update_worker_model(self, worker_model: nn.Module, worker_id: int) -> None:
        self.worker_models[worker_id] = worker_model
        self.update_global_model(worker_model, worker_id)


class CPUAsyncTrainer(AsyncTrainerBase):
    def __init__(
        self,
        model: nn.Module,
        num_workers: int=4,
        staleness_max: int=5,
    ):
        super().__init__(model, num_workers, staleness_max)
        self.worker_models = None

    def initialize_worker_models(self) -> None:
        self.worker_models = [copy.deepcopy(self.model).cpu() for _ in range(self.num_workers)]
        print(f"Initialized {self.num_workers} worker models in CPU memory")

    def get_worker_model(self, worker_id: int) -> nn.Module:
        version_params = self._get_parameter_version(self.worker_versions[worker_id])
        
        with torch.no_grad():
            for name, param in self.worker_models[worker_id].named_parameters():
                if name in version_params:
                    param.data.copy_(version_params[name])
        
        gpu_model = self.worker_models[worker_id].to(self.device)
        
        if self.worker_optimizers[worker_id] is not None:
            self.worker_optimizers[worker_id] = None
        
        return gpu_model
    
    def update_worker_model(self, worker_model: nn.Module, worker_id: int) -> None:
        # Update global first while still on device
        self.update_global_model(worker_model, worker_id)
        
        # Then move to CPU
        self.worker_models[worker_id] = worker_model.cpu()
        
        # Clear optimizer since parameters moved
        if self.worker_optimizers[worker_id] is not None:
            del self.worker_optimizers[worker_id]
            self.worker_optimizers[worker_id] = None


class DiskAsyncTrainer(AsyncTrainerBase):
    def __init__(
        self,
        model: nn.Module,
        num_workers: int=4,
        staleness_max: int=5,
        checkpoint_dir: str="async_checkpoints",
        alpha: float=0.9,  # Make alpha configurable
    ):
        super().__init__(model, num_workers, staleness_max, alpha)
        self.checkpoint_dir = checkpoint_dir
        self.base_model = copy.deepcopy(model)
        
        # Always keep the latest version in memory
        self.latest_state_dict = copy.deepcopy(model.state_dict())
        
        # Track which versions have been saved to disk
        self.saved_versions = set([0])
        
        # Save version threshold - save at most this many versions
        self.max_saved_versions = staleness_max + 5
        
        # Keep a small cache of recent versions in memory
        self.version_cache = {}  # version -> state_dict
        self.version_cache[0] = self.latest_state_dict
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Save initial model
        self._save_version_to_disk(0, self.latest_state_dict)

    def _save_version_to_disk(self, version, state_dict):
        path = os.path.join(self.checkpoint_dir, f"global_v{version}.pt")
        torch.save(state_dict, path)
        self.saved_versions.add(version)
        
        # Clean up old versions if we have too many
        if len(self.saved_versions) > self.max_saved_versions:
            versions_to_keep = sorted(self.saved_versions)[-self.max_saved_versions:]
            for v in list(self.saved_versions):
                if v not in versions_to_keep:
                    old_path = os.path.join(self.checkpoint_dir, f"global_v{v}.pt")
                    if os.path.exists(old_path):
                        os.remove(old_path)
                    self.saved_versions.remove(v)

    def _get_version_from_disk(self, version):
        path = os.path.join(self.checkpoint_dir, f"global_v{version}.pt")
        if os.path.exists(path):
            return torch.load(path)
        return None

    def initialize_worker_models(self) -> None:
        # Reset worker versions
        for worker_id in range(self.num_workers):
            self.worker_versions[worker_id] = 0
        
        print(f"Initialized {self.num_workers} worker models (references) at {self.checkpoint_dir}")

    def _snapshot_parameters(self):
        # Override to save parameters to both memory history and update latest state dict
        super()._snapshot_parameters()
        
        # Update latest state dict
        self.latest_state_dict = copy.deepcopy(self.model.state_dict())
        
        # Add to cache
        self.version_cache[self.global_version] = self.latest_state_dict
        
        # Clean cache if needed
        if len(self.version_cache) > self.max_saved_versions // 2:
            versions_to_keep = sorted(self.version_cache.keys())[-self.max_saved_versions//2:]
            for v in list(self.version_cache.keys()):
                if v not in versions_to_keep and v != 0:  # always keep version 0
                    del self.version_cache[v]
        
        # Save less frequently to disk
        if self.global_version % 5 == 0 or random.random() < 0.1:  # Save periodically or with small chance
            self._save_version_to_disk(self.global_version, self.latest_state_dict)

    def get_worker_model(self, worker_id: int) -> nn.Module:
        # Create a fresh model
        worker_model = copy.deepcopy(self.base_model).to(self.device)
        target_version = self.worker_versions[worker_id]
        
        # Try to load parameters from cache first
        found_in_cache = False
        if target_version in self.version_cache:
            # Load from cache
            state_dict = self.version_cache[target_version]
            worker_model.load_state_dict(state_dict)
            found_in_cache = True
        
        # If not in cache, try to load from disk
        elif target_version in self.saved_versions:
            state_dict = self._get_version_from_disk(target_version)
            if state_dict:
                worker_model.load_state_dict(state_dict)
                # Add to cache for future use
                self.version_cache[target_version] = state_dict
                found_in_cache = True
        
        # If not found in cache or disk, use parameter history
        if not found_in_cache:
            version_params = self._get_parameter_version(target_version)
            with torch.no_grad():
                for name, param in worker_model.named_parameters():
                    if name in version_params:
                        param.data.copy_(version_params[name].to(self.device))
        
        return worker_model

    def update_worker_model(self, worker_model: nn.Module, worker_id: int) -> None:
        # Use parent method for consistency
        self.update_global_model(worker_model, worker_id)
        
        # Cache the latest state dict
        self.latest_state_dict = copy.deepcopy(self.model.state_dict())
        self.version_cache[self.global_version] = self.latest_state_dict
        
        # Save to disk occasionally
        if self.global_version % 5 == 0 or random.random() < 0.1:
            self._save_version_to_disk(self.global_version, self.latest_state_dict)

        # Clean cache if needed
        if len(self.version_cache) > self.max_saved_versions // 2:
            versions_to_keep = sorted(self.version_cache.keys())[-self.max_saved_versions//2:]
            for v in list(self.version_cache.keys()):
                if v not in versions_to_keep and v != 0:  # always keep version 0
                    del self.version_cache[v]


class TrueAsyncTrainer:
    def __init__(
        self,
        model: nn.Module,
        num_workers: int=4,
        staleness_max: int=5,
        alpha: float=0.9,
        optimizer_class=optim.SGD,
        optimizer_kwargs={"lr": 0.01, "momentum": 0.9}
    ):
        self.model = model.share_memory()  
        self.device = device
        self.num_workers = num_workers
        self.staleness_max = staleness_max
        self.alpha = alpha
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        
        self.global_version = mp.Value('i', 0)
        self.stop_signal = mp.Value('b', False)
        self.param_lock = mp.Lock()
        
        self.loss_queue = mp.Queue()
        self.update_queue = mp.Queue()
        
    def worker_process(self, worker_id, dataloader):
        worker_model = copy.deepcopy(self.model).to(self.device)
        optimizer = self.optimizer_class(worker_model.parameters(), **self.optimizer_kwargs)
        
        data_iter = cycle(dataloader)
        
        steps = 0
        while not self.stop_signal.value:
            with self.param_lock:
                current_version = self.global_version.value
                
                staleness = random.randint(0, self.staleness_max)
                worker_version = max(0, current_version - staleness)
                
                with torch.no_grad():
                    for w_param, g_param in zip(worker_model.parameters(), self.model.parameters()):
                        w_param.data.copy_(g_param.data.to(self.device))
            
            local_steps = random.randint(1, 3)  
            
            for _ in range(local_steps):
                data, target = next(data_iter)
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = worker_model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                self.loss_queue.put((steps, worker_id, loss.item()))
                steps += 1
            
            with self.param_lock:
                with torch.no_grad():
                    for g_param, w_param in zip(self.model.parameters(), worker_model.parameters()):
                        g_param.data = self.alpha * g_param.data + (1 - self.alpha) * w_param.cpu().data
                
                self.global_version.value += 1
                
                self.update_queue.put((self.global_version.value, worker_id))
    
    def train(self, dataloader, total_steps=1000):
        processes = []
        
        for worker_id in range(self.num_workers):
            p = mp.Process(
                target=self.worker_process,
                args=(worker_id, dataloader)
            )
            p.start()
            processes.append(p)
        
        pbar = tqdm(total=total_steps, desc="Global Training Progress")
        loss_history = []
        update_history = []
        
        step = 0
        try:
            while step < total_steps:
                while not self.loss_queue.empty():
                    step_info = self.loss_queue.get()
                    loss_history.append(step_info)
                    step += 1
                    if step <= total_steps:
                        pbar.update(1)
                    
                while not self.update_queue.empty():
                    update_info = self.update_queue.get()
                    update_history.append(update_info)
                    tqdm.write(f"Global model updated to v{update_info[0]} by worker {update_info[1]}")
                
                time.sleep(0.01)  
                
        except KeyboardInterrupt:
            print("Training interrupted")
        except Exception as e:
            print(f"Error during training: {e}")
        
        finally:
            self.stop_signal.value = True
            
            for p in processes:
                p.join()
                
            pbar.close()
            
        return loss_history, update_history
    
    def evaluate(self, test_loader):
        """Evaluate the global model on test data"""
        model = self.model.to(self.device)
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        
        return test_loss, accuracy


def create_async_trainer(model, config=None):
    if config is None:
        config = AsyncTrainerConfig()
    
    if config.storage_mode == 'gpu':
        return GPUAsyncTrainer(model, config)
    elif config.storage_mode == 'cpu':
        return CPUAsyncTrainer(model, config)
    elif config.storage_mode == 'disk':
        return DiskAsyncTrainer(model, config)
    elif config.storage_mode == 'true_async':
        return TrueAsyncTrainer(model, config)
    else:
        raise ValueError(f"Unknown storage mode: {config.storage_mode}")

def prepare_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    batch_size = 32
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def main():
    config = AsyncTrainerConfig(
        storage_mode='disk',
        num_workers=4, 
        staleness_max=5,
        total_steps=10000,
        batch_size=32,
        tqdm_update_freq=5,  # Update progress bars every 5 steps
        use_wandb=True,
        wandb_project="async-training-experiment",
        wandb_name="disk-storage-test",
        wandb_log_freq=10  # Log to wandb every 10 steps
    )
    
    model = SimpleCNN().to(device)
    trainer = create_async_trainer(model, config)
    
    train_loader, test_loader = prepare_dataset(batch_size=config.batch_size)
    
    trainer.train_step(train_loader)
    trainer.evaluate(test_loader)
    
    # Close wandb run
    if config.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()