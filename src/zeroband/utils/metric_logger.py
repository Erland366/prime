import importlib
import importlib.util
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from faker import Faker


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
    def __init__(self, project, logger_config): ...

    def log(self, metrics: dict[str, Any]): ...

    def finish(self): ...

class CombineMetricLogger:
    def __init__(self, project, logger_config, resume: bool, logger_cls: list[str]):
        assert isinstance(logger_cls, list)
        self._logger_cls = []

        name = logger_config["config"]["run_name"]
        if name is None:
            random_name = Faker().name()
            random_int = random.randint(0, 100)
            name = random_name.lower().replace(" ", "_") + str(random_int)
            logger_config["config"]["run_name"] = name

        for metric_logger_type in logger_cls:
            if metric_logger_type == "wandb":
                logger_cls = WandbMetricLogger 
            elif metric_logger_type == "tensorboard":
                logger_cls = TensorboardMetricLogger
            else:
                logger_cls = DummyMetricLogger
            metric_logger = logger_cls(
                project=project,
                logger_config=logger_config,
                resume=resume,
            )
            self._logger_cls.append(metric_logger)

    def log(self, metrics: dict[str, Any], step=None):
        for i in range(len(self._logger_cls)):
            self._logger_cls[i].log(metrics, step)

    def finish(self):
        for i in range(len(self._logger_cls)):
            self._logger_cls[i].finish()

class TensorboardMetricLogger:
    def __init__(self, project, logger_config, resume: bool):
        if importlib.util.find_spec("tensorboard") is None:
            raise ImportError("wandb is not installed. Please install it to use WandbMonitor.")

        base_dir = os.getenv("LOG_DIR", "logging")

        from torch.utils.tensorboard import SummaryWriter
        self._step = 0

        # Save config to name
        name = logger_config["config"]["run_name"]

        create_project_folder(os.path.join(base_dir, project, name))

        # TODO: Make sure everything is serializable
        # with open(f"{os.path.join(base_dir, project, name, 'config.json')}", "w") as f:
        #     json.dump(config, f, indent=4)

        self._writer = SummaryWriter(
            log_dir=os.path.join(base_dir, project, name),
            purge_step=None if resume else 0,
            comment=name
        )

    def log(self, metrics: dict[str, Any], step=None):
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, step or self._step)
        self._step += 1

    def finish(self):
        self._writer.close()

class WandbMetricLogger(MetricLogger):
    def __init__(self, project, logger_config, resume: bool):
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is not installed. Please install it to use WandbMonitor.")

        import wandb

        wandb.init(
            project=project, config=logger_config, name=logger_config["config"]["run_name"], resume="auto" if resume else None
        )  # make wandb reuse the same run id if possible

    def log(self, metrics: dict[str, Any], step=None):
        import wandb

        wandb.log(metrics, step)

    def finish(self):
        import wandb

        wandb.finish()


class DummyMetricLogger(MetricLogger):
    def __init__(self, project, logger_config, *args, **kwargs):
        self.project = project
        self.logger_config = logger_config
        open(self.project, "a").close()  # Create an empty file to append to

        self.data = []

    def log(self, metrics: dict[str, Any]):
        self.data.append(metrics)

    def finish(self):
        with open(self.project, "wb") as f:
            pickle.dump(self.data, f)
