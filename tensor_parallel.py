import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional
import process_group_manager as pgm
import math

def split_tensor_along_last_dim(tensor, num_partitions):
    last_dim_size = tensor.size()[-1] // num_partitions
    return torch.split(tensor, last_dim_size, dim=-1)

class Gather(torch.autograd.Function):
    "Gather in forward pass and scatter in backward pass"
    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        input = input.contiguous()
        tensor_list = [torch.empty_like(input) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = input
        dist.all_gather(tensor_list, input, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=-1).contiguous()
        return output
    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        # Split gradient according to TP size
        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()

class Broadcast(torch.autograd.Function):
    "Broadcast in forward pass and all reduce in backward pass"
    @staticmethod
    def forward(ctx, input):
        return input
    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_world_size)
        return grad_output    
    
class Reduce(torch.autograd.Function):
    "All reduce in forward pass and broadcast in backward pass"
    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_world_size)
        return input
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output    
    
def apply_tensor_parallel(model):
    def _replace_module(_module, _linear_proj_name, _style, args={}):
        assert _style in ["column", "row", 'vocab']
        linear_layer = getattr(_module, _linear_proj_name)
        
        if _style == "column":
            new_linear_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                gather_output=args.get("gather_output", False)
            )
        elif _style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
            )
        else:
            new_linear_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings,
                embedding_dim=linear_layer.embedding_dim,
            )
        setattr(_module, _linear_proj_name, new_linear_layer)

    module_linear_name_stype_mapping_list = [
        ("attention", "q_proj", "column"),
        ("attention", "k_proj", "column"),
        ("attention", "v_proj", "column"),
        ("attention", "out_proj", "row"),
        ("mlp", "up_proj", "column"),
        ("mlp", "gate_proj", "column"),
        ("mlp", "down_proj", "row"),
    ]

    for layer in model.decoder_layers:
        for module_name, linear_proj_name, style in module_linear_name_stype_mapping_list:
            _replace_module(getattr(layer, module_name), linear_proj_name, style)
            
    _replace_module(model, "embedding", "vocab")
    _replace_module(model, "final_proj", "column", args={"gather_output": True})
    
    return model

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:bool, gather_output:bool = False):
        super(ColumnParallelLinear).__init__()
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        # Weight matrix will be of shape (out_features, in_features) for a torch.nn.Linear(in_features, out_features)
        # But when we do matrix multiplication with the input features of shape (batch_size, in_features), we multiply it with the transpose of weight matrix
        # that is, of shape (in_features, out_features). We have to split the weight.transpose matrix along the column dimension across multiple GPUs, which is along out_features dimension.
        assert out_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    def reset_parameters(self):
        if self.tp_world_size == 1:
            # Default initialization used in torch.nn.Linear layer
            bound = 1/math.sqrt(self.weight.size(1))
            torch.nn.init.uniform(self.weight, -bound, bound)
            return
        master_weight = torch.empty(self.out_features,self.in_features,dtype=self.weight.dtype,requires_grad=False)
        bound = 1/math.sqrt(master_weight.size(1))
        torch.nn.init.uniform(master_weight, -bound, bound)
        # Split this master weight across GPUs
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()
    def forward(self, input):
        input_parallel = Broadcast.apply(input)
        output = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = Gather.apply(output)
        return output


class RowParallelLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:bool):
        super(RowParallelLinear).__init__()
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features
        # Weight matrix will be of shape (out_features, in_features) for a torch.nn.Linear(in_features, out_features)
        # But when we do matrix multiplication with the input features of shape (batch_size, in_features), we multiply it with the transpose of weight matrix
        # that is, of shape (in_features, out_features). We have to split along the weight.transpose matrix along the row dimension across multiple GPUs, which is along in_features dimension.
        assert self.in_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias term to 0
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.tp_world_size == 1:
            # Default initialization used in torch.nn.Linear layer
            bound = 1/math.sqrt(self.weight.size(1))
            torch.nn.init.uniform(self.weight, -bound, bound)
            return
        master_weight = torch.empty(self.out_features,self.in_features,dtype=self.weight.dtype,requires_grad=False)
        bound = 1/math.sqrt(master_weight.size(1))
        torch.nn.init.uniform(master_weight, -bound, bound)
        # Split this master weight across GPUs
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()
    
    def forward(self, input):
        output_parallel = F.linear(input, self.weight, self.bias)
        output = Reduce.apply(output_parallel)
        return output

class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        super(VocabParallelEmbedding).__init__()
        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, pgm.process_group_manager.tp_rank, pgm.process_group_manager.tp_world_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        self.weight = nn.Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        if self.tp_world_size == 1:
            torch.nn.init.normal(self.weight, mean=0, std=1)
            return 
        master_weight = torch.empty(self.num_embeddings, self.embedding_dim, dtype=self.weight.dtype, requires_grad=False)
        torch.nn.init.normal(master_weight, mean=0, std=1)
        weight_list = torch.split(master_weight, self.num_embeddings_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()
