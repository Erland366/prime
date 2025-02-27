from typing import Iterable

import torch
import torch.distributed.fsdp
import torch.distributed.tensor

from distributed_shampoo import (
    DefaultEigenvalueCorrectedShampooConfig,
    DistributedShampoo,
    FullyShardShampooConfig,
    ShampooPT2CompileConfig,
)

from zeroband.config import Config, AdamConfig, SoapConfig, OptimizersConfig


def get_optimizer(
    config: Config, 
    params: Iterable[torch.nn.Parameter],
    experiment_config: "ExperimentConfig",
) -> torch.optim.Optimizer:
    """
    Obtain the optimizer for the model.
    """

    _config: OptimizersConfig = config.optim.optim

    if isinstance(_config, AdamConfig):
        opt = torch.optim.AdamW(
            params,
            lr=_config.lr,
            weight_decay=_config.weight_decay,
            betas=(_config.betas1, _config.betas2),
            fused=experiment_config.fused_optimizer
>>>>>>> d57965b04574262815a174246707afd9615eed0c
        )
    elif isinstance(_config, SoapConfig):
        opt = DistributedShampoo(
            params,
            lr=_config.lr,
            betas=(_config.betas1, _config.betas2),
            epsilon=1e-12,
            weight_decay=_config.weight_decay,
            max_preconditioner_dim=_config.max_preconditioner_dim,
            precondition_frequency=_config.precondition_frequency,
            use_decoupled_weight_decay=True,
            # This can also be set to `DefaultSOAPConfig` which uses QR decompositions, hence is
            # less expensive and might thereby allow for a smaller `precondition_frequency`.
            preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,
            distributed_config=FullyShardShampooConfig(),
            shampoo_pt2_compile_config=ShampooPT2CompileConfig(
                enable_shampoo_pt2_dynamic_shape=False
            ),
        )
    else:
        raise ValueError(f"Unknown optimizer {_config.optimizer}")

    return opt


__all__ = ["OptimizersConfig", "get_optimizer"]
