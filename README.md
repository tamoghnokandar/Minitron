# Distributed training framework
## Installation
```bash
pip install -e .
```
## How to run
```bash
# 3D parallelism (Tensor + Data + Pipeline parallel)
torchrun --nproc_per_node 8 train.py --tp_size 2 --pp_size 2 --pp_engine 1f1b --dp_size 2 --micro_batch_size 2 --gradient_accumulation_steps 8 --seq_len 1024 --max_tokens 4096000 --num_proc 16 --model_name TinyLlama/TinyLlama_v1.1 --num_hidden_layers 22 --num_attention_heads 32 --num_key_value_heads 4 --run_name 3D_parallelism_1B --use_wandb
****
