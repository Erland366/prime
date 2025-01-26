# Testing

# Data

# Testing Command
```
GLOO_SOCKET_IFNAME=lo GLOBAL_ADDR=localhost GLOBAL_RANK=0 GLOBAL_UNIQUE_ID=0 GLOBAL_WORLD_SIZE=1 GLOBAL_PORT=8989  torchrun --nproc_per_node=2 src/zeroband/train.py  @configs/150M_short/A40.toml
```

```
GLOO_SOCKET_IFNAME=lo GLOBAL_ADDR=localhost GLOBAL_RANK=0 GLOBAL_UNIQUE_ID=0 GLOBAL_WORLD_SIZE=1 GLOBAL_PORT=8989  torchrun --nproc_per_node=2 src/zeroband/train.py  @configs/my_configs/A100.toml
```

# Debug
GLOO_SOCKET_IFNAME=lo GLOBAL_ADDR=localhost GLOBAL_RANK=0 GLOBAL_UNIQUE_ID=0 GLOBAL_WORLD_SIZE=1 GLOBAL_PORT=8989  debugpy-run -m torch.distributed.run -- --nproc_per_node=2 src/zeroband/train.py  @configs/150M_short/A40.toml

```
GLOO_SOCKET_IFNAME=lo GLOBAL_ADDR=localhost GLOBAL_RANK=0 GLOBAL_UNIQUE_ID=0 GLOBAL_WORLD_SIZE=1 GLOBAL_PORT=8989  debugpy-run -m torch.distributed.run -- --nproc_per_node=2 src/zeroband/train.py  @configs/my_configs/A100.toml
```

# Simulate DiLoCo
```
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node_diloco.sh 2 1 src/zeroband/train.py  @configs/my_configs/without_fsdp.toml
```