import os
import torch
import torch.distributed as dist

class ProcessGroupManager:
    def __init__(self, dp_size, pp_size, tp_size): 
        """"
        Example 2 nodes each consisting of 4 GPUs will have global_ranks - [0, 1, 2, 3, 4, 5, 6, 7]
        and local_ranks will be for Node 1 - [0, 1, 2, 3] and for Node 2 - [0, 1, 2, 3] 
        """
        self.global_rank = dist.get_rank() # ID of the GPU of the current process
        self.world_size = dist.get_world_size() # Total number of GPUs
        self.local_rank = int(os.environ["LOCAL_RANK"])
        assert self.world_size == dp_size*pp_size*tp_size , f"World size ({self.world_size}) != DP ({self.dp_size}) * PP ({self.pp_size}) * TP ({self.tp_size})"
        
        """Build a 3D grid of ranks"""
        self.grid = torch.arange(self.world_size).view(dp_size, pp_size, tp_size)

        """Find the indices of grid for this particular process using nonzero()"""
        self.dp_rank, self.pp_rank, self.tp_rank = (self.grid == self.global_rank).nonzero().flatten().tolist() 
        
        """Process Group Creation - Creates a new process group object in PyTorch’s distributed backend. The [0] at the end means: return the subgroup corresponding to this process.
        So this gives you the communication handle for the respective parallelism operations.
        """
        self.tp_group = dist.new_subgroups_by_enumeration([self.grid[d, p, :].tolist() for d in range(dp_size) for p in range(pp_size)])
        self.pp_group = dist.new_subgroups_by_enumeration([self.grid[d, :, t].tolist() for d in range(dp_size) for t in range(tp_size)])[0]
        self.dp_group = dist.new_subgroups_by_enumeration([self.grid[:, p, t].tolist() for p in range(pp_size) for t in range(tp_size)])[0]
        self.pp_dp_group = dist.new_subgroups_by_enumeration([self.grid[:, :, t].flatten().tolist() for t in range(tp_size)])[0]
        
        """The default global group of all processes."""
        self.world_group = dist.group.WORLD


        """Update group IDs with new grid ordering. 
        This line just stores the rank IDs (integers) that belong to each process group.
        If dp_size=tp_size=pp_size=2 for example and global_rank = 6 then dp_rank=1, pp_rank=1 and tp_rank=0.
        tensor([[[0, 1],
         [2, 3]],

        [[4, 5],
         [6, 7]]])
        The self.tp_group_ids = [6, 7] in this case.
        """
        self.tp_group_ids = self.grid[self.dp_rank, self.pp_rank, :].tolist()
        self.pp_group_ids = self.grid[self.dp_rank, :, self.tp_rank].tolist()
        self.dp_group_ids = self.grid[:, self.pp_rank, self.tp_rank].tolist()
        
        # Data Parallelism
        self.dp_world_size = dist.get_world_size(group=self.dp_group)
        self.dp_first_rank = self.dp_group_ids[0]
        self.dp_last_rank = self.dp_group_ids[-1]

        # Pipeline Parallelism
        self.pp_world_size = dist.get_world_size(group=self.pp_group)
        self.pp_first_rank = self.pp_group_ids[0]
        self.pp_last_rank = self.pp_group_ids[-1]
        self.pp_is_first_stage = self.pp_rank == 0
        self.pp_is_last_stage = self.pp_rank == self.pp_world_size - 1
        self.pp_next_rank = None if self.pp_rank == self.pp_world_size - 1 else int(self.grid[self.dp_rank, self.pp_rank + 1, self.tp_rank].item())
        self.pp_prev_rank = None if self.pp_rank == 0 else int(self.grid[self.dp_rank, self.pp_rank - 1, self.tp_rank].item())

        # Tensor Parallelism
        self.tp_world_size = dist.get_world_size(group=self.tp_group)
        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]
    
    """Makes printing an object show a nice summary."""
    def __str__(self):
        return f"DP({self.dp_world_size})-PP({self.pp_world_size})-TP({self.tp_world_size})-Rank({self.global_rank})"

def setup_process_group_manager(dp_size, pp_size, tp_size):
    global process_group_manager
    process_group_manager = ProcessGroupManager(dp_size, pp_size, tp_size)
        

