import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from Dataset import ERA5Dataset, ERA5DatasetDistributed


def get_local_groups_fourway(rank):
    """
    Return the group for pairs of ranks.
    
    Groups are for pairs that share the same variables
    Ex. if there are 8 total ranks, will return a list of groups [0,2], [1,3], [4,6], [5,7]
    """
    gpus_per_node = 4 # horeka- specific
    
    n_groups = torch.distributed.get_world_size() // gpus_per_node
    local_groups = []

    for node in range(n_groups):
        base = node * gpus_per_node
        local_groups.append(
            torch.distributed.new_group(ranks=[base, base + 2])
        )
        local_groups.append(
            torch.distributed.new_group(ranks=[base + 1, base + 3])
        )

    return local_groups

def get_ddp_groups(parallelism):
    """
    Return the corresponding DDP groups.

    Ex. if there are 8 total ranks, will return a list of groups [0,4]
    """
    n_nodes = torch.distributed.get_world_size() // parallelism
    # 
    global_group_list = []
    global_groups     = []

    for node in np.arange(parallelism):
        global_group_list.append(((np.arange(n_nodes)*parallelism+node)).tolist())
               
    for global_group in global_group_list:
        global_groups.append(torch.distributed.new_group(ranks=global_group))
    
    return global_groups

def get_local_groups_twoway(rank):
    """
    Return a group per model-parallel instance.

    Only used for 2-way parallel. 
    """
    # [0,1] --> one group
    # [2,3] --> another group
    local_groups = []
    

    for i in range(torch.distributed.get_world_size() // 2):
        # 8 --> [0, 4]
        local_pair_start_rank = i * 2
        local_groups.append(
                torch.distributed.new_group(ranks=[local_pair_start_rank, local_pair_start_rank+1]
                                            )
            )
    
    return local_groups[rank//2]

def get_local_group_per_node(rank):
    """
    Return a group per model-parallel instance.

    Only used for 4-way parallel. 
    """
    # [0, 1, 2, 3]
    # Rank 4, 6 should be in group together
    local_groups = []
    for i in range(torch.distributed.get_world_size() // 4):
        local_pair_start_rank = i * 4
        local_groups.append(
                torch.distributed.new_group(
                    ranks=[local_pair_start_rank, local_pair_start_rank+1, local_pair_start_rank+2, local_pair_start_rank+3]
                )
        )
    
    return local_groups[rank//4]

def set_all_seeds(seed, rank=None):
    """
    Initialize random seeds.

    Set random seeds with different ranks getting different values.
    """
    if rank is None:
        rank = 0
    os.environ["PL_GLOBAL_SEED"] = str(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

def set_all_seeds_local_rank(seed, rank=None, parallelism=4):
    """
    Set local group seeds.

    Set seeds with all ranks in the same model-parallel instance to the same value.
    """
    if rank is None:
        rank = 0
    os.environ["PL_GLOBAL_SEED"] = str(seed + rank//parallelism) 
    torch.manual_seed(seed + rank//parallelism)
    torch.cuda.manual_seed_all(seed + rank//parallelism)


def get_dataloader(config_dict, data_path, mode='train', subset_size=0, distributed=False, batch_size=1, normalize=True, rank=None, dt=None):
    """
    Return a dataloader.

    config_dict: yaml
    data_path: String
        path to data
    mode: String
        valid values are 'train' or 'valid'
    subset_size: int
        Size of subset if desired. Else, if subset is 0, take the entire dataset
    distributed: Boolean
        flag for data parallel
    batch_size: int
    normalize: Boolean
        flag to indicate if data in dataloader should be normalized
    rank: int
        rank of process
    dt: int
        lead time [hours], if the value should override the config file.
    """
    parallelism = config_dict['model_parallel']['parallelism']
    dt = config_dict['data']['dt'] if dt is None else dt
    if parallelism == 1:
        dataset = ERA5Dataset(config_dict, data_path, mode=mode, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize)
    elif parallelism == 4 and config_dict['model_parallel']['split'] == 1:
        dataset = ERA5Dataset(config_dict, data_path, mode=mode, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, rank=rank)    
    else:
        dataset = ERA5DatasetDistributed(config_dict, data_path, mode=mode, rank=rank, forecast_length=config_dict['model']['latent_rollout'], normalize=normalize, dt=dt)
    
    if subset_size != 0: 
        subset_indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, subset_indices)            


    sampler    = DistributedSampler(dataset, 
                                    shuffle=True,
                                    num_replicas=(torch.distributed.get_world_size()//parallelism),
                                    rank=rank//parallelism
                                    ) 
    
    if mode == 'validation' and parallelism > 1:
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                sampler=sampler, 
                                drop_last=True, 
                                pin_memory=torch.cuda.is_available(), 
                                prefetch_factor=config_dict['data']['prefetch_factor'], 
                                num_workers=config_dict['data']['num_workers']
                                ) 
        return dataloader
    else:
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=(sampler is None), 
                                sampler=sampler, 
                                drop_last=True, 
                                pin_memory=torch.cuda.is_available(), 
                                prefetch_factor=config_dict['data']['prefetch_factor'], 
                                num_workers=config_dict['data']['num_workers']
                                )

    return dataloader

def init_CUDA():
    """
    Initialize DistributedDataParallel .
    
    Returns
    -------
    device: String
    slurm_localid: int
    rank: int
    world_size: int
    """
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
    slurm_localid = int(os.getenv("SLURM_LOCALID"))

    print(f"Rank {rank}: world_size {world_size}, slurm_localid {slurm_localid}")
    # Initialize GPUs and dataloaders
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(slurm_localid)
    
    # Initialize DistributedDataParallel.
    dist.init_process_group(backend="nccl", 
                            rank=rank, 
                            world_size=world_size, 
                            init_method="env://")
    
    if dist.is_initialized(): 
        print(f"Rank {rank}/{world_size}: Process group initialized with torch rank {torch.distributed.get_rank()} and torch world size {torch.distributed.get_world_size()}.")
    else:
        print(f"Rank {rank} not initialized")

    
    return device, slurm_localid, rank, world_size

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Per-process.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_constant_mask_2d(
    patch_size,
    folder_path="/hkfs/work/workspace/scratch/ke4365-summa/constant_masks/",
    xlat=721
):
    """
    Load the constant masks applied in the patch embedding layer.

    patch_size: Tuple(int, int)
        Number of pixels in (lat, lon) dimensions per patch
    folder_path: String
        Path to directory containing constant masks

    Returns
    -------
    land_mask: Tensor
        of shape (n_lat, n_lon) after padding
    soil_type: Tensor
        of shape (n_lat, n_lon) after padding
    topography: Tensor
        of shape (n_lat, n_lon) after padding
    """
    # Load data from numpy files
    data_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    data = {}
    for file in data_files:
        file_path = os.path.join(folder_path, file)
        data[file] = np.load(file_path)

    soil_type = data["soil_type.npy"]
    topography = data["topography.npy"]

    soil_type = (soil_type - np.mean(soil_type)) / np.std(soil_type)
    topography = (topography - np.mean(topography)) / np.std(topography)
    # Torch tensors
    land_mask = torch.tensor(data["land_mask.npy"]).to(torch.float32)
    soil_type = torch.tensor(soil_type).to(torch.float32)
    topography = torch.tensor(topography).to(torch.float32)

    # Check that the shapes of all the data are the same
    assert (
        land_mask.shape == soil_type.shape == topography.shape
    ), "Shapes of the three constant masks are not equal."

    land_mask = land_mask[:xlat]
    soil_type = soil_type[:xlat]
    topography = topography[:xlat]
    
    return land_mask, soil_type, topography

def plot_fields(input, output, target, save_path, index=1):
    """
    Plot and save the model input, output - input, and target - input.

    input: Tensor
        of shape (n_lat, n_lon, vars)
    output: Tensor
        of shape (n_lat, n_lon, vars)
    target: Tensor
        of shape (n_lat, n_lon, vars)
    save_path: String
        path to save visualizations to
    index: int
        specifices the index of the variable to visualize
    """
    input_print = input[index].detach().cpu()
    output_print = output[index].detach().cpu()
    target_print = target[index].detach().cpu()

    vmin = target_print.min()
    vmax = target_print.max()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    c0 = ax0.pcolormesh(input_print)
    fig.colorbar(c0, ax=ax0)
    c1 = ax1.pcolormesh(output_print, vmin=vmin, vmax=vmax)
    fig.colorbar(c1, ax=ax1)
    c2 = ax2.pcolormesh(target_print, vmin=vmin, vmax=vmax)
    fig.colorbar(c2, ax=ax2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# save model
def save_model(model, optimizer, file_path, rank, epoch):
    """
    Save model, optimizer state, epoch to file.

    model: Pytorch model
    optimizer: instance of optimizer class
    file_path: String
        directory to save to
    rank: int
        rank of process
    epoch: int
    """
    file_path = f"{file_path}/model_{rank}.pt"
    state = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, file_path)

def load_model(model, optimizer, file_path, rank, model_parallel=False, parallelism=2):
    """
    Load model from checkpoint.

    model: Pytorch model
    optimizer: optimizer state dicts
    file_path: String
        directory to load model from
    model_parallel: Boolean
        flag for whether Jigsaw tensor parallelism is used
    parallelism: int
        if parallel, number of processes the model is distributed over
    """
    if not model_parallel:
        rank = 0
    if model_parallel:
        load_rank = rank % parallelism
        file_path = f'{file_path}/model_{load_rank}.pt'
    state = torch.load(f"{file_path}", map_location='cpu')
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])

    epoch = state['epoch']
    return model, optimizer, epoch


def load_and_push_data(data, device, non_blocking=False):
    """
    Read input data and pushes to device. 
    
    Returns [data, False]
    """
    pushed_data = data.to(device, non_blocking=non_blocking)
    return pushed_data 

        
def reduce_ddp_layernorms(parallelism, mlp, local_groups, global_groups, rank, node_rank):
    """
    Reduce gradients across local groups for 4-way parallel.
    
    For processes that predict the same variables.
    """
    # Now manually all-reduce each parameter's grad
    handles = []
    gradients = [param.grad for name, param in mlp.named_parameters() if (param.grad is not None and 'layer_norm' in name)]
    flat_grad = parameters_to_vector(gradients)
    handles = torch.distributed.all_reduce(
        flat_grad, 
        op=torch.distributed.ReduceOp.AVG,
        group=local_groups[rank % 2 + node_rank*2],
        async_op=True
    )
            
    return handles, flat_grad, gradients