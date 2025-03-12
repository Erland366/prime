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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        num_workers: int=4,
        staleness_max: int=5,
        alpha: float=0.9,
    ):
        self.model = model
        self.device = device
        self.num_workers = num_workers
        self.staleness_max = staleness_max
        self.alpha = alpha

        self.global_version = 0
        self.worker_versions = [0] * num_workers

        self.param_history = {}
        self.param_history_depth = staleness_max + 2

        for name, param in self.model.named_parameters():
            self.param_history[name] = deque(maxlen=self.param_history_depth)
            self.param_history[name].append((0, param.data.clone()))

        self.worker_active = [False] * num_workers
        self.worker_positions = [0] * num_workers
        self.loss_history = []
        self.worker_history = []
        
        # Create optimizers for each worker
        self.worker_optimizers = [None] * num_workers

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
        
        # Track worker history
        self.worker_history.append((self.global_version, worker_id))
        
        tqdm.write(f"Global model updated to version {self.global_version} by worker {worker_id}")
    
    def train_step(self, dataloader, total_steps=100, optimizer_class=optim.SGD, 
                  optimizer_kwargs={"lr": 0.01, "momentum": 0.9}):
        # Create a pool of worker processes
        step = 0
        self.initialize_worker_models()
        
        # Create persistent data loaders (using cycle to continue iteration)
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

                # Check if worker needs initialization
                if self.worker_optimizers[worker_id] is None:
                    self.worker_optimizers[worker_id] = optimizer_class(
                        worker_model.parameters(), **optimizer_kwargs
                    )

                worker_device = next(worker_model.parameters()).device
                
                # Mark worker as active
                if not self.worker_active[worker_id]:
                    self.worker_active[worker_id] = True
                    
                    # Simulate staleness - worker gets a version of parameters
                    staleness = random.randint(0, self.staleness_max)
                    target_version = max(0, self.global_version - staleness)
    
                    self.worker_versions[worker_id] = target_version
                    worker_pbars[worker_id].set_postfix({
                        "version": target_version, 
                        "loss": 0.0,
                        "active": "Yes"
                    })
                    worker_pbars[worker_id].refresh()
    
                # Get batch from the worker's dataloader
                data, target = next(worker_dataloaders[worker_id])
                data, target = data.to(self.device), target.to(self.device)
                self.worker_positions[worker_id] += 1
                
                # Update progress bar for this worker
                worker_pbars[worker_id].total = self.worker_positions[worker_id]
                worker_pbars[worker_id].n = self.worker_positions[worker_id] - 1
                worker_pbars[worker_id].refresh()
                
                # Get optimizer for this worker
                optimizer = self.worker_optimizers[worker_id]

                # Ensure all tensors in optimizer state are on the right device
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.device != worker_device:
                            # This shouldn't happen but let's be safe
                            raise RuntimeError(f"Parameter device mismatch: {p.device} vs {worker_device}")

                # Compute gradients and update worker model
                optimizer.zero_grad()
                loss = self._compute_loss(worker_model, data, target)
                loss.backward()
                optimizer.step()
                # Track loss
                self.loss_history.append((step, worker_id, loss.item()))
                
                # Update progress bar with loss information
                current_postfix = worker_pbars[worker_id].postfix if isinstance(worker_pbars[worker_id].postfix, dict) else {}
                current_postfix["loss"] = f"{loss.item():.4f}"
                worker_pbars[worker_id].set_postfix(current_postfix)
                
                # Store previous global version to check if it was updated
                prev_global_version = self.global_version
                
                # Update worker model and potentially update global model
                self.update_worker_model(worker_model, worker_id)
                
                # Check if global version was incremented and update progress bar accordingly
                if self.global_version > prev_global_version:
                    global_pbar.n = self.global_version
                    global_pbar.refresh()
                
                # Simulate worker completion
                if random.random() < 0.2:  # 20% chance of worker finishing its task
                    self.worker_active[worker_id] = False
                    worker_pbars[worker_id].set_postfix({
                        "version": self.worker_versions[worker_id], 
                        "loss": f"{loss.item():.4f}",
                        "active": "No"
                    })

                if step % 20 == 0 and step > 0:
                    # breakpoint()
                    pass
                
                step += 1
                main_pbar.update(1)
                
        finally:
            # Close all progress bars
            main_pbar.close()
            for pbar in worker_pbars.values():
                pbar.close()
            global_pbar.close()
            
            # Clean up resources
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
        
        return test_loss, accuracy
    
    def cleanup(self):
        """Clean up resources"""
        # Clear parameter history
        for name in self.param_history:
            self.param_history[name].clear()
        
        # Delete optimizers
        for i in range(len(self.worker_optimizers)):
            if self.worker_optimizers[i] is not None:
                del self.worker_optimizers[i]
                self.worker_optimizers[i] = None


class GPUAsyncTrainer(AsyncTrainerBase):
    def __init__(
        self,
        model: nn.Module,
        num_workers: int=4,
        staleness_max: int=5,
    ):
        super().__init__(model, num_workers, staleness_max)
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
        # Update the worker model first
        self.worker_models[worker_id] = worker_model
        # Then update the global model
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
        # Get parameters for this version
        version_params = self._get_parameter_version(self.worker_versions[worker_id])
        
        # Apply parameters to CPU model first
        with torch.no_grad():
            for name, param in self.worker_models[worker_id].named_parameters():
                if name in version_params:
                    param.data.copy_(version_params[name])
        
        # Then move to device and return
        return self.worker_models[worker_id].to(self.device)

    def update_worker_model(self, worker_model: nn.Module, worker_id: int) -> None:
        # First update the global model while the worker model is still on the device
        self.update_global_model(worker_model, worker_id)
        
        # Now handle moving model to CPU and updating optimizer
        if self.worker_optimizers[worker_id] is not None:
            # Get the current optimizer state before moving the model
            optimizer_state = self.worker_optimizers[worker_id].state_dict()
            
            # Move all tensors in optimizer state to CPU first
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cpu()
            
            # Now move model to CPU
            cpu_model = worker_model.cpu()
            self.worker_models[worker_id] = cpu_model
            
            # Re-create optimizer with CPU model
            self.worker_optimizers[worker_id] = type(self.worker_optimizers[worker_id])(
                cpu_model.parameters(),
                **{k: v for k, v in self.worker_optimizers[worker_id].defaults.items()}
            )
            
            # Load the CPU optimizer state
            self.worker_optimizers[worker_id].load_state_dict(optimizer_state)
        else:
            # Just move the model to CPU
            self.worker_models[worker_id] = worker_model.cpu()


class DiskAsyncTrainer(AsyncTrainerBase):
    def __init__(
        self,
        model: nn.Module,
        num_workers: int=4,
        staleness_max: int=5,
        checkpoint_dir: str="async_checkpoints",
    ):
        super().__init__(model, num_workers, staleness_max)
        self.checkpoint_dir = checkpoint_dir
        self.base_model = copy.deepcopy(model)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def initialize_worker_models(self) -> None:
        initial_state = self.model.cpu().state_dict()

        for worker_id in range(self.num_workers):
            path = os.path.join(self.checkpoint_dir, f"worker_{worker_id}_v{self.worker_versions[worker_id]}.pt")
            torch.save(initial_state, path)
        
        print(f"Initialized {self.num_workers} worker models on disk at {self.checkpoint_dir}")

    def get_worker_model(self, worker_id: int) -> nn.Module:
        version_params = self._get_parameter_version(self.worker_versions[worker_id])

        worker_model = copy.deepcopy(self.base_model).to(self.device)

        with torch.no_grad():
            for name, param in worker_model.named_parameters():
                if name in version_params:
                    param.data.copy_(version_params[name].to(self.device))

        return worker_model

    def update_worker_model(self, worker_model: nn.Module, worker_id: int) -> None:
        # Save both model and optimizer state to disk
        model_path = os.path.join(self.checkpoint_dir, f"worker_{worker_id}_v{self.global_version}_model.pt")
        optim_path = os.path.join(self.checkpoint_dir, f"worker_{worker_id}_v{self.global_version}_optim.pt")
        
        # Save model state
        torch.save(worker_model.cpu().state_dict(), model_path)
        
        # Save optimizer state if it exists
        if self.worker_optimizers[worker_id] is not None:
            torch.save(self.worker_optimizers[worker_id].state_dict(), optim_path)
        
        # Clean up old version
        old_model_path = os.path.join(self.checkpoint_dir, f"worker_{worker_id}_v{self.worker_versions[worker_id]}_model.pt")
        old_optim_path = os.path.join(self.checkpoint_dir, f"worker_{worker_id}_v{self.worker_versions[worker_id]}_optim.pt")
        
        for old_path in [old_model_path, old_optim_path]:
            if os.path.exists(old_path) and old_path != model_path:
                os.remove(old_path)
                
        # Then update the global model
        self.update_global_model(worker_model, worker_id)
    
    def cleanup(self):
        """Clean up resources including disk files"""
        super().cleanup()
        
        # Remove all checkpoint files
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("worker_") and filename.endsWith(".pt"):
                os.remove(os.path.join(self.checkpoint_dir, filename))


# Implement true asynchronous training using multiprocessing
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
        self.model = model.share_memory()  # Enable model sharing between processes
        self.device = device
        self.num_workers = num_workers
        self.staleness_max = staleness_max
        self.alpha = alpha
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        
        # Shared variables for synchronization
        self.global_version = mp.Value('i', 0)
        self.stop_signal = mp.Value('b', False)
        self.param_lock = mp.Lock()
        
        # For tracking metrics
        self.loss_queue = mp.Queue()
        self.update_queue = mp.Queue()
        
    def worker_process(self, worker_id, dataloader):
        worker_model = copy.deepcopy(self.model).to(self.device)
        optimizer = self.optimizer_class(worker_model.parameters(), **self.optimizer_kwargs)
        
        # Create cyclic dataloader
        data_iter = cycle(dataloader)
        
        steps = 0
        while not self.stop_signal.value:
            # Get current global version
            with self.param_lock:
                current_version = self.global_version.value
                
                # Apply staleness (lag behind global model)
                staleness = random.randint(0, self.staleness_max)
                worker_version = max(0, current_version - staleness)
                
                # Apply global model parameters to worker model
                with torch.no_grad():
                    for w_param, g_param in zip(worker_model.parameters(), self.model.parameters()):
                        w_param.data.copy_(g_param.data.to(self.device))
            
            # Train for a few local steps
            local_steps = random.randint(1, 3)  # Do 1-3 local updates
            
            for _ in range(local_steps):
                data, target = next(data_iter)
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = worker_model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                # Report loss
                self.loss_queue.put((steps, worker_id, loss.item()))
                steps += 1
            
            # Update global model
            with self.param_lock:
                # Apply updates to global model with alpha smoothing
                with torch.no_grad():
                    for g_param, w_param in zip(self.model.parameters(), worker_model.parameters()):
                        g_param.data = self.alpha * g_param.data + (1 - self.alpha) * w_param.cpu().data
                
                # Increment global version
                self.global_version.value += 1
                
                # Report update
                self.update_queue.put((self.global_version.value, worker_id))
    
    def train(self, dataloader, total_steps=1000):
        processes = []
        
        # Launch worker processes
        for worker_id in range(self.num_workers):
            p = mp.Process(
                target=self.worker_process,
                args=(worker_id, dataloader)
            )
            p.start()
            processes.append(p)
        
        # Track progress
        pbar = tqdm(total=total_steps, desc="Global Training Progress")
        loss_history = []
        update_history = []
        
        step = 0
        try:
            while step < total_steps:
                # Process losses
                while not self.loss_queue.empty():
                    step_info = self.loss_queue.get()
                    loss_history.append(step_info)
                    step += 1
                    if step <= total_steps:
                        pbar.update(1)
                    
                # Process updates
                while not self.update_queue.empty():
                    update_info = self.update_queue.get()
                    update_history.append(update_info)
                    tqdm.write(f"Global model updated to v{update_info[0]} by worker {update_info[1]}")
                
                time.sleep(0.01)  # Small sleep to avoid busy waiting
                
        except KeyboardInterrupt:
            print("Training interrupted")
        except Exception as e:
            print(f"Error during training: {e}")
        
        finally:
            # Signal workers to stop
            self.stop_signal.value = True
            
            # Wait for processes to finish
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


def create_async_trainer(model, storage_mode='cpu', num_workers=4, staleness_max=5, **kwargs):
    if storage_mode == 'gpu':
        return GPUAsyncTrainer(model, num_workers, staleness_max)
    elif storage_mode == 'cpu':
        return CPUAsyncTrainer(model, num_workers, staleness_max)
    elif storage_mode == 'disk':
        checkpoint_dir = kwargs.get('checkpoint_dir', 'async_checkpoints')
        return DiskAsyncTrainer(model, num_workers, staleness_max, checkpoint_dir)
    elif storage_mode == 'true_async':
        optimizer_class = kwargs.get('optimizer_class', optim.SGD)
        optimizer_kwargs = kwargs.get('optimizer_kwargs', {"lr": 0.01, "momentum": 0.9})
        return TrueAsyncTrainer(model, num_workers, staleness_max, 
                               alpha=kwargs.get('alpha', 0.9),
                               optimizer_class=optimizer_class,
                               optimizer_kwargs=optimizer_kwargs)
    else:
        raise ValueError(f"Unknown storage mode: {storage_mode}")

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
    model = SimpleCNN().to(device)

    trainer = create_async_trainer(model, storage_mode='cpu', num_workers=1, staleness_max=5)

    train_loader, test_loader = prepare_dataset()

    trainer.train_step(train_loader, total_steps=10000)

    trainer.evaluate(test_loader)

if __name__ == '__main__':
    main()