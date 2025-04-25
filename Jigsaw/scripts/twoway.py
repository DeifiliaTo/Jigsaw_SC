
import numpy as np
import torch
import torch.distributed
import torch.optim as optim
import os
import sys
sys.path.insert(1, f'{os.getenv("SRC")}/')
sys.path.insert(1, f'{os.getenv("SRC")}/src')
import utils, eval
import Model
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import yaml
import helper

def training_loop(config_dict, device, localid, rank):   
    model_parallel = config_dict['model']['model_parallel']
    parallelism = config_dict['model_parallel']['parallelism'] if model_parallel else 1
    
    utils.set_all_seeds(1, rank=(rank%parallelism)) if parallelism > 1 else utils.set_all_seeds(1, rank=None) # ensure ranks 0, 4, ...8... have the same initial parameters
    
    train_subset, valid_subset = config_dict['data']['train_subset'], config_dict['data']['valid_subset']
    train_subset = None if train_subset < 0 else train_subset
    valid_subset  = None if valid_subset < 0 else valid_subset
    if rank == 0:
        print(f'Training parameters {config_dict["training"]}')
        print(f'Model parameters {config_dict["model"]}')
        print(f'Training subset size {train_subset}')
        print(f'Data parameters {config_dict["data"]}')
    
    best_val_loss = 1e5
    
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
    
    local_group = utils.get_local_groups_twoway(rank)
    local_groups = None
    node_groups = local_group
    global_groups = utils.get_ddp_groups(parallelism)

    mlp = Model.Model_twoway(config_dict, device, rank, local_group).to(device)

    mlp = DDP(
        mlp,
        device_ids=[localid],
        output_device=localid,
        process_group=global_groups[rank % parallelism]
    )

    lr = config_dict['training']['lr']
    
    optimizer = optim.Adam([
            {'params': mlp.module.params.embedding.parameters(), 'lr': config_dict['training']['lr_embedding']},
            {'params': mlp.module.params.recovery.parameters(), 'lr': config_dict['training']['lr_recovery']},
            {'params': mlp.module.params.common.parameters()}
            ], 
            lr=lr, 
            weight_decay=3e-6
    )

    
    utils.set_all_seeds_local_rank(1, rank=rank, parallelism=parallelism) # Set all GPUs to the same seed so that they read the same data           
    
    train_dataloader = utils.get_dataloader(config_dict, config_dict['data']['train_data_path'], mode='train', distributed=(not model_parallel), batch_size=config_dict['training']['train_batch'], subset_size=train_subset, rank=rank)
    
    if rank == 0:
        print(f"model has {utils.count_parameters(mlp)*parallelism/1e6} million parameters")
        flop_forward, comm_forward = helper.calc_flops_comm(config_dict)
        print(f'Estimated forward TFLOPs and GB_COMM: {flop_forward}, {comm_forward}')

    # load
    try:
        file_path = config_dict['training']['load_path']
        mlp, optimizer, epoch = utils.load_model(mlp, optimizer=optimizer, file_path=file_path, rank=rank, model_parallel=model_parallel)
        epoch = epoch + 1
        
    except Exception as err:
        print(err)
        epoch = 0
        if rank == 0:
            print("Initializing model from scratch")
        pass
    
    
    mlp = mlp.to(device)

    loss_fn = torch.nn.MSELoss()
    
    start_lr = 1e-6
    if config_dict['training']['tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    epoch_times = np.zeros(11)
    buffer, buffer2 = None, None

    torch.cuda.synchronize()

    for epoch in range(epoch, 11):
        train_dataloader.sampler.set_epoch(epoch)

        mlp.train()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i, data in enumerate(train_dataloader):          
            # Loading data: Always with async ops and handles
            input_data_or_handle  = utils.load_and_push_data(data[0], device)
            target_data_or_handle = utils.load_and_push_data(data[1][0], device)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            input_data = input_data_or_handle
            
            output = mlp(input_data, rollout_=0+1) # output shape [n_batch, 720, 720, 36]
                    
            target_data = target_data_or_handle
            
            # Reshape output
            output = (output.reshape(output.shape[0], xlat, xlon, 36))             
            
            loss = loss_fn(output, target_data)
            
            loss.backward()
            
        end.record()
        torch.cuda.synchronize()

        if rank == 0:
            epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60 
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")

    if rank == 0:
        print(f'Average epoch time is {np.average(epoch_times[1:])}')
        print(f'Std epoch time is {np.std(epoch_times[1:])}')
        

if __name__ == '__main__':
    model_parallel = False
    train_subset = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    device, localid, rank, world_size = utils.init_CUDA()

    
    training_loop(config_dict, device, localid, rank)
    torch.distributed.destroy_process_group()
