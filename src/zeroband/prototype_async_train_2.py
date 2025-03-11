import concurrent.futures as futures
import copy
import dataclasses
import math
import numpy as np
import random
import time
import torch
import torch.nn as nn

from concurrent import futures
from typing import Optional, Any
from torch.utils.data import DataLoader
from transformers import set_seed

set_seed(42)
device='cuda' if torch.cuda.is_available() else 'cpu'

@dataclasses.dataclass
class TrainConfig:
    ep = .5
    clusters_per_class = 8
    d = 512
    num_classes = 4
    batch_size = 32
    num_data = 128000  
    num_data_small = 4096
    num_workers = 4
    num_data_shards = 4
    data_mode = 'big'

train_config = TrainConfig()

@dataclasses.dataclass
class Worker:
    worker_id: int
    device_sec_per_step: float
    model_id: Optional[int] = None
    data_id: Optional[int] = None
    num_train_steps: Optional[int] = None
    future: Optional[futures.Future[Any]] = None
    training_finished_time: float = 0.0

@dataclasses.dataclass
class Data:
    data_id: int
    visit_times: int = 0
    local_updates: int = 0

def sample_train_task(worker, experiment, train_steps, scale_steps=False):
    data_id = sample_data(experiment.data_shards)
    worker.data_id = data_id
    experiment.data_shards[data_id].visit_times += 1

    if not scale_steps:
        worker.num_train_steps = train_steps
    else:  # Let slower worker run fewer steps to remove staleness
        c = min([w.device_sec_per_step for w in workers]) / worker.device_sec_per_step
        worker.num_train_steps = max(int(train_steps * c), 5)

    worker.training_finished_time += worker.num_train_steps * worker.device_sec_per_step
    experiment.data_shards[data_id].local_updates += worker.num_train_steps
    worker.model_id = experiment.model_id
    return data_id

def sample_data(data_shards):
    # We sample the data shard that has been visited least.
    visit_times = np.array([d.visit_times for d in data_shards])
    min_visit = np.min(visit_times)
    min_indices = np.where(visit_times == min_visit)[0]
    data_id = np.random.choice(min_indices)
    return data_shards[data_id].data_id

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.Tensor(X)
        self.Y = torch.LongTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y

class CosineScheduler:
    def __init__(self, optimizer, peak_lr, T_max, last_epoch=-1, eta_min=1e-6):
        self.last_epoch = last_epoch
        self.total_epochs = T_max
        self.peak_lr = peak_lr
        self.eta_min = eta_min
        self.optimizer = optimizer

    def step(self):
        self.last_epoch += 1
        progress = min(self.last_epoch / self.total_epochs, 1.0)
        lr = self.eta_min + (self.peak_lr-self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def get_optimizer(params, opt_name, opt_kwargs):
    opt = {
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'sgd_momentum': torch.optim.SGD,
        'nesterov': torch.optim.SGD,
    }
    return opt[opt_name](params, **opt_kwargs)

class Model(nn.Module):
    def __init__(self, d, hidden_size=65000, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class Experiment:
    def __init__(
        self,
        data_mode: str = 'big',
        model_hidden_size=65000,
        default_inner_steps=20,
        total_phases=400,
        eval_every=10,
        lr_schedule='cosine',
        seed=42,
        inner_opt_name='adamw',
        outer_opt_name='nesterov',
        inner_opt_kwargs={'lr': 2e-4, 'weight_decay': 1e-3, 'betas': (0.9, 0.999)},
        outer_opt_kwargs={'lr': 0.1, 'nesterov': True, 'momentum': 0.9},
        verbose=False,
        x=None,
        y=None,
        cluster_indices_iid=None,
        eval_indices=None,
    ):
        self.default_inner_steps = default_inner_steps
        self.eval_interval = self.default_inner_steps * train_config.num_workers * eval_every
        self.total_phases = total_phases
        self.max_local_updates = self.default_inner_steps * train_config.num_workers * total_phases
        self.max_local_updates_per_data = self.default_inner_steps * total_phases
        self.eval_target_step = 0

        self.inner_opt_name = inner_opt_name
        self.inner_opt_kwargs = inner_opt_kwargs
        self.outer_opt_name = outer_opt_name
        self.outer_opt_kwargs = outer_opt_kwargs

        self.verbose = verbose
        self.sync_time = 0.0
        self.lr_schedule = lr_schedule
        self.model_hidden_size = model_hidden_size
        self.data_mode = data_mode
        self.seed = seed
        assert x is not None and y is not None and cluster_indices_iid is not None and eval_indices is not None
        self.initialize(x, y, eval_indices, cluster_indices_iid)

    def get_experiment_name(self):
        opt_string = f"{self.outer_opt_name}-{self.inner_opt_name}"
        return opt_string

    @property
    def finished(self):
        # We terminate if the total steps across all workers is beyond threshold
        return self.num_total_local_updates >= self.max_local_updates

    def initialize(self, x, y, eval_indices, cluster_indices_iid):
        set_seed(self.seed)
        print(f"\n[info] Starting experiment {self.get_experiment_name()} ...",
                flush=True)
        self.model_id = 0
        self.time_now = 0.0
        self.sync_start_time = None
        self.num_total_local_updates = 0
        self.stats = []  # Keep track of train/eval stats

        # Define workers
        sec_per_steps = [0.58, 0.75, 1.51, 3.33]
        self.workers  = [Worker(i, s) for i, s in enumerate(sec_per_steps)]

        # Define data shards
        if self.data_mode == 'small':
            eval_dataset = Dataset(x_[eval_indices_small], y_[eval_indices_small])
            datasets = [Dataset(x_[iid], y_[iid]) for iid in cluster_indices_iid_small]
        elif self.data_mode == 'big':
            eval_dataset = Dataset(x[eval_indices], y[eval_indices])
            datasets = [Dataset(x[iid], y[iid]) for iid in cluster_indices_iid]
        else:
            raise ValueError(f"Unknown data mode: {self.data_mode}")

        self.data_shards = [Data(i) for i in range(train_config.num_data_shards)]

        def get_train_indices(dataset):
            n = len(dataset)
            repeat_times = max(self.max_local_updates_per_data*train_config.batch_size // n * 2, 1)
            indices = []
            for i in range(repeat_times):
                x = np.arange(n)
                random.shuffle(x)
                indices.append(x)
            indices = np.concatenate(indices)
            return indices

        self.datasets = datasets
        # train_indices are for controlling randomness, and data loading
        self.train_indices = [
            get_train_indices(dataset) for dataset in self.datasets
        ]
        self.eval_data_loader = DataLoader(eval_dataset,
                                            batch_size=256,
                                            shuffle=False)
        # Define task writer
        self.train_writer = futures.ThreadPoolExecutor(max_workers=train_config.num_workers)
        # Define the model on the server
        self.server_model = Model(hidden_size=self.model_hidden_size, d=train_config.d, num_classes=train_config.num_classes)
        self.server_model.to(device)

        # Define the outer optimizer
        self.server_optimizer = get_optimizer(self.server_model.parameters(),
                                                self.outer_opt_name,
                                                self.outer_opt_kwargs)

    def close(self):
        futures.wait([w.future for w in self.workers if w.future and not w.future.done()])
        self.train_writer.shutdown()
        del self.eval_data_loader
        del self.server_model
        del self.server_optimizer

    def evaluate(self):
        if self.num_total_local_updates < min(self.eval_target_step, self.max_local_updates):
            return
        self.eval_target_step += self.eval_interval

        self.server_model.eval()
        correct = 0
        test_loss = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        counter = 0
        with torch.no_grad():
            for data, target in self.eval_data_loader:
                data, target = data.to(device), target.to(device)  # Move data to GPU
                output = self.server_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                counter += data.shape[0]
        test_loss /= counter
        test_accuracy = correct / counter
        # Record the test statistics
        self.stats.append((self.num_total_local_updates,
                            test_loss,
                            test_accuracy,
                            self.time_now))
        if self.verbose:
            print(f"[info] step {self.num_total_local_updates:6d} | "
                f"time {self.time_now:10.2f} | acc {test_accuracy:4.2f} | "
                f"loss {test_loss:.6f}", flush=True)

def inner_loop(
    local_model,
    dataset,
    train_indices,
    train_steps,
    lr_schedule,
    local_updates,
    total_updates,
    opt_name,
    opt_kwargs
):
    local_model.train()
    initial_state = [p.clone().data for p in local_model.parameters()]

    optimizer = get_optimizer(local_model.parameters(), opt_name, opt_kwargs)

    if lr_schedule == 'cosine':
        lr_scheduler = CosineScheduler(
            optimizer,
            peak_lr=2e-4,
            T_max=total_updates,
            last_epoch=local_updates - train_steps,
            eta_min=1e-6,
        )
    else:
        lr_scheduler = None

    step = 0
    criterion = nn.CrossEntropyLoss()

    
    begin_updates = local_updates - train_steps
    for idx in range(begin_updates, local_updates):
        indices = train_indices[idx*train_config.batch_size:(idx+1)*train_config.batch_size]
        data, target = dataset.X[indices], dataset.Y[indices]
        data = torch.Tensor(data).to(device)
        target = torch.LongTensor(target).to(device)
        optimizer.zero_grad()
        output = local_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if lr_scheduler: lr_scheduler.step()

    pseudo_gradient = [
        (init - final.data).clone().data
        for init, final in zip(initial_state, local_model.parameters())
    ]
    return {'pseudo_gradient': pseudo_gradient}

class SyncExperiment(Experiment):
    def __init__(self, record_cossim_every=20, record_cossim=False, **kwargs):
        self.record_cossim_every = record_cossim_every
        self.record_cossim = record_cossim
        self.cossims = []
        super().__init__(**kwargs)

    def get_experiment_name(self):
        string = super().get_experiment_name()
        return f"sync-{string}"

    def synchronize(self, pseudo_gradients):
        for w in self.workers:
            self.time_now = max(self.time_now, w.training_finished_time)
            self.num_total_local_updates += w.num_train_steps
            self.time_now += self.sync_time

        
        sync_gradient = [sum(vs) / train_config.train_config.num_workers for vs in zip(*pseudo_gradients)]

        
        self.server_optimizer.zero_grad()
        for p, g in zip(self.server_model.parameters(), sync_gradient):
            p.grad = g.data.clone()
        self.server_optimizer.step()

        for w in self.workers:
            w.training_finished_time = self.time_now  
        self.model_id += 1

    def train(self):
        pseudo_gradients = []
        for w in self.workers:
            data_id = sample_train_task(w,
                                        self,
                                        self.default_inner_steps)
            local_model = copy.deepcopy(self.server_model)
            w.future = self.train_writer.submit(
                inner_loop,
                local_model,
                self.datasets[data_id],
                self.train_indices[data_id],
                self.default_inner_steps,
                self.lr_schedule,
                self.data_shards[data_id].local_updates,
                self.max_local_updates_per_data,
                self.inner_opt_name,
                self.inner_opt_kwargs
            )

        pseudo_gradients = [w.future.result()['pseudo_gradient'] for w in self.workers]
        return pseudo_gradients

    def cossim(self, pseudo_gradients):
        pgs = torch.vstack(
            [
            torch.cat([x.view(-1) for x in pseudo_gradient])
            for pseudo_gradient in pseudo_gradients
            ]
        )
        normed = pgs / pgs.norm(dim=1)[:, None]
        sim_matrix = torch.mm(normed, normed.t())
        sim_matrix_np = sim_matrix.cpu().numpy()
        return sim_matrix_np

    def run(self):
        self.evaluate()

        while not self.finished:
            pseudo_gradients = self.train()
            if self.model_id % self.record_cossim_every == 0 and self.record_cossim:
                self.cossims.append((self.num_total_local_updates, self.cossim(pseudo_gradients)))
                self.synchronize(pseudo_gradients)
                self.evaluate()

        self.close()

class AsyncExperiment(Experiment):
    def __init__(self, sync_method='vanilla', sync_weight_method='constant', max_wait_time=0.5, **kwargs):
        self.sync_method = sync_method
        self.sync_weight_method = sync_weight_method
        self.max_wait_time = max_wait_time
        super().__init__(**kwargs)

    def get_experiment_name(self):
        string = super().get_experiment_name()
        if self.max_wait_time > 0:
            return f"async-{string}-sync_method={self.sync_method}-weight_method={self.sync_weight_method}"
        else:
            return f"async-{string}-sync_method={self.sync_method}-weight_method={self.sync_weight_method}-nowait"

    def synchronizable(self):
        return (not self.sync_start_time) or (
            self.time_now - self.sync_start_time <= self.max_wait_time
        )

    def get_next_sync_worker(self):
        if all(not w.future for w in self.workers): return None  

        wid = np.argmin(
            np.array([
                w.training_finished_time if w.future else np.inf
                for w in self.workers
            ])
        )
        w = self.workers[wid]
        if (
            self.sync_start_time
            and (w.training_finished_time - self.sync_start_time)
            > self.max_wait_time
        ):  
            return None
        return w

    def get_sync_weight(self, staleness):
        K = len(self.workers)
        base = np.sqrt(K) / K
        if self.sync_weight_method == 'constant':
            return base
        elif self.sync_weight_method == 'polynomial':
            return base * 1 / (1 + staleness)**0.5
        else:
            raise ValueError(f'Unknown sync_weight_method: {self.sync_weight_method}')

    def vanilla(self, pseudo_gradient, sync_weight):
        self.server_optimizer.zero_grad()
        for p, g in zip(self.server_model.parameters(), pseudo_gradient):
            p.grad = sync_weight * g.data.clone()
        self.server_optimizer.step()

    def delayed_sgd(self, pseudo_gradient, sync_weight):
        if self.model_id % train_config.train_config.num_workers == 0:
            self.server_optimizer.zero_grad()
        for p, g in zip(self.server_model.parameters(), pseudo_gradient):
            if p.grad is None:
                p.grad = (1/train_config.train_config.num_workers) * g.data.clone()
            else:
                p.grad += (1/train_config.train_config.num_workers) * g.data.clone()
        if (self.model_id + 1) % train_config.train_config.num_workers == 0:
            self.server_optimizer.step()

    def synchronize(self, w):
        self.sync_start_time = self.sync_start_time or self.time_now
        staleness = (self.model_id - w.model_id)
        self.num_total_local_updates += w.num_train_steps
        
        sync_weight = self.get_sync_weight(staleness)
        pseudo_gradient = w.future.result()['pseudo_gradient']

        if self.sync_method == 'vanilla':
            self.vanilla(pseudo_gradient, sync_weight)
        elif self.sync_method == 'delayed_sgd':
            self.delayed_sgd(pseudo_gradient, sync_weight)

        self.model_id += 1
        self.time_now += self.sync_time
        w.future = None  
        for w_ in self.workers:
            if w_.future is None: w_.training_finished_time = self.time_now

    def train(self):
        for w in self.workers:
            if w.future is None:
                data_id = sample_train_task(w,
                                            self,
                                            self.default_inner_steps)
                local_model = copy.deepcopy(self.server_model)
                w.future = self.train_writer.submit(
                    inner_loop,
                    local_model,
                    self.datasets[data_id],
                    self.train_indices[data_id],
                    self.default_inner_steps,
                    self.lr_schedule,
                    self.data_shards[data_id].local_updates,
                    self.max_local_updates_per_data,
                    self.inner_opt_name,
                    self.inner_opt_kwargs
                )

    def run(self):
        
        self.evaluate()
        self.train()

        while not self.finished:
            next_worker = self.get_next_sync_worker()

            if next_worker:
                futures.wait([next_worker.future])
                assert next_worker.future.done()
                self.time_now = max(self.time_now, next_worker.training_finished_time)

            if next_worker and self.synchronizable():
                self.synchronize(next_worker)
                self.evaluate()
            else:
                self.sync_start_time = None
                self.train()

        self.close()

class Experiment:
    def __init__(
        self,
        data_mode: str = 'big',
        model_hidden_size=65000,
        default_inner_steps=20,
        total_phases=400,
        eval_every=10,
        lr_schedule='cosine',
        seed=42,
        inner_opt_name='adamw',
        outer_opt_name='nesterov',
        inner_opt_kwargs={'lr': 2e-4, 'weight_decay': 1e-3, 'betas': (0.9, 0.999)},
        outer_opt_kwargs={'lr': 0.1, 'nesterov': True, 'momentum': 0.9},
        verbose=False
    ):
        self.default_inner_steps = default_inner_steps
        self.eval_interval = self.default_inner_steps * train_config.train_config.num_workers * eval_every
        self.total_phases = total_phases
        self.max_local_updates = self.default_inner_steps * train_config.train_config.num_workers * total_phases
        self.max_local_updates_per_data = self.default_inner_steps * total_phases
        self.eval_target_step = 0

        self.inner_opt_name = inner_opt_name
        self.inner_opt_kwargs = inner_opt_kwargs
        self.outer_opt_name = outer_opt_name
        self.outer_opt_kwargs = outer_opt_kwargs

        self.verbose = verbose
        self.sync_time = 0.0
        self.lr_schedule = lr_schedule
        self.model_hidden_size = model_hidden_size
        self.data_mode = data_mode
        self.seed = seed
        self.initialize()

    def get_experiment_name(self):
        opt_string = f"{self.outer_opt_name}-{self.inner_opt_name}"
        return opt_string

    @property
    def finished(self):
        
        return self.num_total_local_updates >= self.max_local_updates

    def initialize(self, x, y, eval_indices, cluster_indices_iid):
        set_seed(self.seed)
        print(f"\n[info] Starting experiment {self.get_experiment_name()} ...",
                flush=True)
        self.model_id = 0
        self.time_now = 0.0
        self.sync_start_time = None
        self.num_total_local_updates = 0
        self.stats = []  

        sec_per_steps = [0.58, 0.75, 1.51, 3.33]
        self.workers  = [Worker(i, s) for i, s in enumerate(sec_per_steps)]

        # Define data shards
        if self.data_mode == 'small':
            eval_dataset = Dataset(X_[eval_indices_small], Y_[eval_indices_small])
            datasets = [Dataset(X_[iid], Y_[iid]) for iid in cluster_indices_iid_small]
        elif self.data_mode == 'big':
            eval_dataset = Dataset(X[eval_indices], Y[eval_indices])
            datasets = [Dataset(X[iid], Y[iid]) for iid in cluster_indices_iid]
        else:
            raise ValueError(f"Unknown data mode: {self.data_mode}")

        self.data_shards = [Data(i) for i in range(train_config.num_data_shards)]

        def get_train_indices(dataset):
            n = len(dataset)
            repeat_times = max(self.max_local_updates_per_data*train_config.batch_size // n * 2, 1)
            indices = []
            for i in range(repeat_times):
                x = np.arange(n)
                random.shuffle(x)
                indices.append(x)
            indices = np.concatenate(indices)
            return indices

        self.datasets = datasets
        # train_indices are for controlling randomness, and data loading
        self.train_indices = [
            get_train_indices(dataset) for dataset in self.datasets
        ]
        self.eval_data_loader = DataLoader(eval_dataset,
                                    batch_size=256,
                                    shuffle=False)
        # Define task writer
        self.train_writer = futures.ThreadPoolExecutor(max_workers=train_config.train_config.num_workers)
        # Define the model on the server
        self.server_model = Model(hidden_size=self.model_hidden_size)
        self.server_model.to(device)

        # Define the outer optimizer
        self.server_optimizer = get_optimizer(
            self.server_model.parameters(),
            self.outer_opt_name,
            self.outer_opt_kwargs
        )

    def close(self):
        futures.wait([w.future for w in self.workers if w.future and not w.future.done()])
        self.train_writer.shutdown()
        del self.eval_data_loader
        del self.server_model
        del self.server_optimizer

    def evaluate(self):
        if self.num_total_local_updates < min(self.eval_target_step, self.max_local_updates):
            return
        self.eval_target_step += self.eval_interval

        self.server_model.eval()
        correct = 0
        test_loss = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        counter = 0
        with torch.no_grad():
            for data, target in self.eval_data_loader:
                data, target = data.to(device), target.to(device)  # Move data to GPU
                output = self.server_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                counter += data.shape[0]
        test_loss /= counter
        test_accuracy = correct / counter
        # Record the test statistics
        self.stats.append(
            (
                self.num_total_local_updates,
                test_loss,
                test_accuracy,
                self.time_now
            )
        )
        if self.verbose:
            print(f"[info] step {self.num_total_local_updates:6d} | "
                f"time {self.time_now:10.2f} | acc {test_accuracy:4.2f} | "
                f"loss {test_loss:.6f}", flush=True)

def make_data(
    N, 
    clusters_per_class, 
    num_classes
):
    # DATA consists of a mixture of CLUSTERS_PER_CLASS*NUM_CLASSES Gaussians. Each
    # Gaussian is assigned to one of the NUM_CLASSES classes; they are distributed
    # amongst the NUM_MODELS clusters evenly. Here, N is the total number of
    # generated examples.
    K = clusters_per_class*num_classes
    n_per_cluster = N//K
    n_per_class = n_per_cluster * clusters_per_class
    centers = np.random.randn(K, train_config.d)
    centers = centers/np.linalg.norm(centers, axis=1, keepdims=True)
    X = np.concatenate([train_config.ep*np.random.randn(n_per_cluster, train_config.d) + centers[i].reshape(1, -1) for i in range(K)])
    Y = np.concatenate([i*np.ones(n_per_class, dtype='int32') for i in range(num_classes)])
    return X, Y

def main():



    if train_config.data_mode == 'small':
        x, y = make_data(train_config.num_data_small, 1, train_config.num_classes)
        total_indices = list(range(len(x)))
        random.shuffle(total_indices)

        cluster_size = len(total_indices) // train_config.num_data_shards
        cluster_indices_iid = [
            total_indices[i*cluster_size: (i+1)*cluster_size]
            for i in range(train_config.num_data_shards)
    ]
        eval_indices = list(range(len(x)))
        eval_indices = random.sample(eval_indices, 10000)
        random.shuffle(eval_indices)
    else:
        x, y = make_data(train_config.num_data, train_config.clusters_per_class, train_config.num_classes)
        total_indices = list(range(len(x)))
        random.shuffle(total_indices)
        cluster_size = len(total_indices) // train_config.num_data_shards
        cluster_indices_iid = [total_indices[i*cluster_size: (i+1)*cluster_size] for i in range(train_config.num_data_shards)]
        eval_indices = list(range(len(x)))
        random.shuffle(eval_indices)
        
    outer_opt_name='nesterov'  
    outer_lr=0.07 
    outer_momentum=0.9 


    outer_opt_kwargs = dict(lr=outer_lr)
    if outer_opt_name in ["nesterov", "sgd", "sgd_momentum"]:
        outer_opt_kwargs["momentum"] = outer_momentum
    else:
        outer_opt_kwargs["betas"] = (outer_momentum, 0.999)

    
    inner_opt_name="adamw"  
    inner_lr=2e-4 
    inner_momentum=0.9 
    inner_weight_decay=0.01 

    inner_opt_kwargs = dict(lr=inner_lr, weight_decay=inner_weight_decay)
    if inner_opt_name in ["nesterov", "sgd", "sgd_momentum"]:
        inner_opt_kwargs["momentum"] = inner_momentum
    else:
        inner_opt_kwargs["betas"] = (inner_momentum, 0.999)

    
    eval_every=10  
    total_phases=100  
    max_wait_time=0.5 
    sync_method='vanilla'  
    sync_weight_method='constant'  
    verbose=True 

    exp = AsyncExperiment(
        outer_opt_name=outer_opt_name,
        outer_opt_kwargs=outer_opt_kwargs,
        inner_opt_name=inner_opt_name,
        inner_opt_kwargs=inner_opt_kwargs,
        max_wait_time=max_wait_time,
        verbose=verbose,
        eval_every=eval_every,
        total_phases=total_phases,
        sync_method=sync_method,
        x=x,
        y=y,
        cluster_indices_iid=cluster_indices_iid,
        eval_indices=eval_indices
    )
    exp.run()


if __name__ == '__main__':
    main()