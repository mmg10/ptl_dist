# PyTorch Lightning for Distributed Training

This repo contains sample code for distributed training for different configurations.

## Distributed Data Parallel
* Data used: MNIST

### Multi-Node, Single GPUs
- [Code](./01_ddp_mnist_2n1g) 
- T4 GPU on 2 instances
Command:
```bash
MASTER_ADDR={IP of RANK 0} MASTER_PORT=29500 NODE_RANK=0 python main.py
MASTER_ADDR={IP of RANK 0} MASTER_PORT=29500 NODE_RANK=1 python main.py
```

### Single-Node, Multiple GPUs
[Code](./02_ddp_mnist_1n2g)   
Command:
```bash
pip install torch_tb_profiler if profiling needed; else disable it
MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=2 NODE_RANK=0 python main.py
tensorboard --logdir=./tensorboard/ --host=0.0.0.0 # to view tensorboard
```

## Distributed Model Parallel
* Data used: Intel Image Classification from 
### Strategy: FSDP

#### Multi-Node, Single GPUs
[Code](./03_fsdp_mnist_2n1g) 
- ViT Model is used  
Caveats:
- Number of devices = 2 (1 per node; 2 in total). This is in contrast with DDP training.
- PTL checkpointing doesn't work. The weights are not stored.
- Models are saved manually and then the best model is copied to the root folder.
```bash
export MASTER_PORT=29500
export MASTER_ADDR=172.31.10.239
export WORLD_SIZE=2
export NODE_RANK=0 # and 1 respectively

python -m torch.distributed.run \
    --nnodes=$WORLD_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK \
    main.py
```

#### Single Node, Multiple GPUs
[Code](./04_fsdp_mnist_1n2g)   
```
MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 NODE_RANK=0 python main.py
```