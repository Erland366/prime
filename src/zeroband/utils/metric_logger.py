import os
import pickle
import random
import json

from faker import Faker
from typing import Any, Protocol, Callable, Optional
from pathlib import Path
import importlib

def create_project_folder(folder_path):
    folder_path = Path(folder_path)
    if folder_path.exists():
        suffix = 1
        while True:
            new_folder_path = Path(f"{folder_path}_{suffix}")
            if not new_folder_path.exists():
                folder_path = new_folder_path
                break
            suffix += 1

    folder_path.mkdir(parents=True, exist_ok=True)

class MetricLogger(Protocol):
    def __init__(self, project, config): ...

    def log(self, metrics: dict[str, Any]): ...

    def finish(self): ...

class CombineMetricLogger:
    def __init__(self, project, config, resume: bool, name: Optional[str], logger_cls: list[str]):
        assert isinstance(logger_cls, list)
        self._logger_cls = []

        if name is None:
            random_name = Faker().name()
            random_int = random.randint(0, 100)
            name = random_name.lower().replace(" ", "_") + str(random_int)

        for metric_logger_type in logger_cls:
            if metric_logger_type == "wandb":
                logger_cls = WandbMetricLogger 
            elif metric_logger_type == "tensorboard":
                logger_cls = TensorboardMetricLogger
            else:
                logger_cls = DummyMetricLogger
            metric_logger = logger_cls(
                project=project,
                config=config,
                resume=resume,
                name=name
            )
            self._logger_cls.append(metric_logger)

    def log(self, metrics: dict[str, Any]):
        for i in range(len(self._logger_cls)):
            self._logger_cls[i].log(metrics)

    def finish(self):
        for i in range(len(self._logger_cls)):
            self._logger_cls[i].finish()

class TensorboardMetricLogger:
    def __init__(self, project, config, resume: bool, name: Optional[str] = None):
        if importlib.util.find_spec("tensorboard") is None:
            raise ImportError("wandb is not installed. Please install it to use WandbMonitor.")

        base_dir = os.getenv("LOG_DIR", "logging")

        from torch.utils.tensorboard import SummaryWriter
        self._step = 0

        # Save config to name
        if name is None:
            random_name = Faker().name()
            random_int = random.randint(0, 100)
            name = random_name.lower().replace(" ", "_") + str(random_int)

        create_project_folder(os.path.join(base_dir, project, name))

        # TODO: Make sure everything is serializable
        # with open(f"{os.path.join(base_dir, project, name, 'config.json')}", "w") as f:
        #     json.dump(config, f, indent=4)

        self._writer = SummaryWriter(
            log_dir=os.path.join(base_dir, project, name),
            purge_step=None if resume else 0,
            comment=name
        )

    def log(self, metrics: dict[str, Any]):
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, self._step)
        self._step += 1

    def finish(self):
        self._writer.close()

class WandbMetricLogger:
    def __init__(self, project, config, resume: bool, name: Optional[str] = None):
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is not installed. Please install it to use WandbMonitor.")

        import wandb

        wandb.init(
            project=project, config=config, resume="auto" if resume else None,
            name=name
        )  # make wandb reuse the same run id if possible

    def log(self, metrics: dict[str, Any]):
        import wandb

        wandb.log(metrics)

    def finish(self):
        import wandb

        wandb.finish()


class DummyMetricLogger:
    def __init__(self, project, config, *args, **kwargs):
        self.project = project
        self.config = config
        open(project, "a").close()  # Create an empty file at the project path

        self.data = []

    def log(self, metrics: dict[str, Any]):
        self.data.append(metrics)

    def finish(self):
        with open(self.project, "wb") as f:
            pickle.dump(self.data, f)
