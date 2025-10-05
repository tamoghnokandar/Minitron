import os 
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
from torch.optim import AdamW
from transformers import AutoConfig

from model import Llama
import process_group_manager as pgm
from process_group_manager import setup_process_group_manager
from utils import set_all_seed, print

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for llama model")

    #Environment Arguments
    parser.add_argument("--omp_num_threads", type=str, default=1) # Set to 1 per GPU in order to avoid thread competition between processes
    parser.add_argument("--tokenizer_parallelism", type=str, default=False) # Set to false in order to avoid too many threads during distributed training

    # Model Arguments
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=4)

    # Dataset Arguments
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=4)

    # Training Arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1e6)

    # Distributed Training Arguments
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--dp_size", type=int, default=1, help="Data Parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--pp_engine", type=str, default="afab", choices=["1f1b", "afab"])

    # Logging arguments
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()
    
    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
    os.environ["TOKENIZER_PARALLELISM"] = args.tokenizer_parallelism
    os.environ["DEVICE"] = "cuda"

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    dist.init_process_group(
    rank=global_rank,           # This process's unique ID
    world_size=world_size,      # Total number of processes
    backend=backend,            # Communication backend (nccl for GPUs)
    init_method="env://",       # Use environment variables for coordination
    timeout=datetime.timedelta(minutes=2)  # Max time to wait for all processes
)

    setup_process_group_manager(dp_size=args.dp_size,pp_size=args.pp_size,tp_size=args.tp_size)


    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_key_value_heads = args.num_key_value_heads
    model_config.max_position_embeddings = args.seq_len

    model = Llama(config=model_config)

    if pgm.process_group_manager.tp_world_size > 1:
        model = apply_tensor_parallel(model)
    
    if pgm.process_group_manager.pp_world_size > 1:
        model = PipelineParallel(model, model_config)
    
    # Need to move the model to the device before wrapping it with DataParallel.
    # Otherwise, the hook will get attached to the CPU model and not the GPU model.
    model.to(dtype).to(device)

    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallelBucket(model)

    model.train()
    dist.barrier()

    # Create dataloader
    dataloader = MicroBatchDataLoader(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_acc_steps=args.gradient_accumulation_steps,
        dataset_name=args.dataset_name,
        tokenizer_name=args.model_name,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
        seed=args.seed,
    )


