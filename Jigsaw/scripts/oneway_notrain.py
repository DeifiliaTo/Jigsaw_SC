
import numpy as np
import torch
import torch.distributed
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
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
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import helper

def training_loop(config_dict, device, localid, rank):
    model_parallel = config_dict['model']['model_parallel']
    parallelism = config_dict['model_parallel']['parallelism'] if model_parallel else 1
    
    utils.set_all_seeds(1, rank=(rank%parallelism)) if parallelism > 1 else utils.set_all_seeds(1, rank=None) # ensure ranks 0, 4, ...8... have the same initial parameters
    
    train_subset, valid_subset = config_dict['data']['train_subset'], config_dict['data']['valid_subset']
    train_subset = None if train_subset < 0 else train_subset
    valid_subset    = None if valid_subset < 0 else valid_subset
            
    best_val_loss = 1e5
    
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']

    area_weights = torch.tensor(eval.calc_weight(xlat)).view(1, xlat, 1, 1).to(device)

    # Define model
    mlp = Model.ModelSequential(config_dict, device, rank).to(device)
    mlp = DDP(
        mlp,
        device_ids=[localid],
        output_device=localid
    )    
        
    lr = config_dict['training']['lr']
    save_path = config_dict['training']['save_dir']
    print_on = config_dict['training']['print']
    
    optimizer = optim.Adam([
            {'params': mlp.module.params.embedding.parameters(), 'lr': config_dict['training']['lr_embedding']},
            {'params': mlp.module.params.recovery.parameters(), 'lr': config_dict['training']['lr_recovery']},
            {'params': mlp.module.params.common.parameters()}
            ], 
            lr=lr, 
            weight_decay=3e-6
    )

    utils.set_all_seeds(1, rank=rank)
    
    train_dataloader = utils.get_dataloader(config_dict, config_dict['data']['train_data_path'], mode='train', distributed=(not model_parallel), batch_size=config_dict['training']['train_batch'], subset_size=train_subset, rank=rank)
        
    if rank == 0:
        print(f"model has {utils.count_parameters(mlp)*parallelism/1e6} million parameters")
        flop_forward, _ = helper.calc_flops_comm(config_dict)
        print(f'Estimated forward TFLOPs: {flop_forward}')

    mlp = mlp.to(device)

    loss_fn = torch.nn.MSELoss()
    
    if config_dict['training']['lr_constant']:
        scheduler = StepLRScheduler(optimizer, decay_rate=1., decay_t=100)
    else:
        scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-5, cycle_decay=0.5, cycle_mul=1, cycle_limit=10)
    scheduler.step(epoch)
    
    start_lr = 1e-6

    if config_dict['training']['tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    epoch_times = np.zeros(11)

    for epoch in range(11):
        train_dataloader.sampler.set_epoch(epoch)

        mlp.train()
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        for i, data in enumerate(train_dataloader):
            # learning rate ramp-up in first iteration
            optimizer.zero_grad() # zero gradient
            
            input_data  = data[0].to(device, non_blocking=False)
            target_data = data[1][0].to(device, non_blocking=False)
                        

        end.record()
        torch.cuda.synchronize()

        if rank == 0:
            epoch_times[epoch] = start.elapsed_time(end) / 1000 / 60 
            print(f"Epoch {epoch} time [min]: {(epoch_times[epoch])}")
        
        scheduler.step(epoch)    
    if rank == 0:
        print(f'Average epoch time is {np.average(epoch_times)}')
        print(f'Std epoch time is {np.std(epoch_times)}')

        
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
