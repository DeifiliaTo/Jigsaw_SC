import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import utils


def calc_weight(n_lat, cossum=458.36551167):
    """
    Return latitude-weighted values for loss function.

    n_lat: int
            number of patches in the latitude dimension
    
    Returns
    -------
    weight: np.array(float)
        latitude_based weighting factor
    """
    latitude = np.linspace(np.pi/2.0, -np.pi/2.0, n_lat)
    weight = n_lat * np.cos(latitude) / cossum
    return weight

# upper_variable options: [Z, Q, T, U, V]
# surface_variable options: [MSLP, U10, V10, T2M]
# PL options: [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
upper_variable_indexing = {'Z': 0, 'Q': 1, 'T': 2, 'U': 3, 'V': 4}
surface_variable_indexing = {'MSL': 0, 'U10': 1, 'V10': 2, 'T2M': 3}
PL_indexing = {1000: 0, 925: 1, 850: 2, 700: 3, 600: 4, 500: 5, 400: 6, 300: 7, 250: 8, 200: 9, 150: 10, 100: 11, 50: 12}


def get_var_names(prepend_variable=['epoch']):
    surface_variables = ['msl', 'u10', 'v10', 't2m']
    pressure_variables = ['z', 'q', 't', 'u', 'v']
    pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    var_names = prepend_variable # lis
    var_names.extend(surface_variables)
    for pvar in pressure_variables:
        var_names.extend(pvar + str(pl) for pl in pressure_levels)
    return var_names
    
def gen_plots(config_dict, input_data, output, target_data, rank, file_path, epoch, model_parallel=False):
    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
    parallelism = config_dict['model_parallel']['parallelism']
    
    if parallelism == 1:
        input_print  = input_data[0].reshape(xlat, xlon, 69).permute(2, 0, 1)
        output_print = output[0].reshape(xlat, xlon, 69).permute(2, 0, 1)
        target_print = target_data[0].reshape(xlat, xlon, 69).permute(2, 0, 1)

        save_path = f"{file_path}/u10_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=1)
        save_path = f"{file_path}/T2M_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=3)
        save_path = f"{file_path}/T850_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=32)
        save_path = f"{file_path}/U850_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=45)
        save_path = f"{file_path}/Z500_{epoch}.png"
        utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=9)
    elif parallelism == 2:
        input_print, output_print, target_print = input_data, output, target_data
        input_print  = input_print.reshape(xlat,  xlon, 36).permute(2, 0, 1)
        output_print = output_print.reshape(xlat, xlon, 36).permute(2, 0, 1)
        target_print = target_print.reshape(xlat, xlon, 36).permute(2, 0, 1)
        
        if rank == 0:
            save_path = f"{file_path}/u10_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=1)
            save_path = f"{file_path}/T2M_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=3)
            save_path = f"{file_path}/T850_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=32)
            save_path = f"{file_path}/Z500_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=9)
        if rank == 1:
            save_path = f"{file_path}/U850_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=45-36)
        
    elif parallelism == 4:
        input_print, output_print, target_print = input_data, output, target_data
        input_print  = input_print.reshape(xlat,  xlon, 69).permute(2, 0, 1)
        output_print = output_print.reshape(xlat, xlon, 69).permute(2, 0, 1)
        target_print = target_print.reshape(xlat, xlon, 69).permute(2, 0, 1)
        
        if rank == 0:
            save_path = f"{file_path}/u10_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=1)
            save_path = f"{file_path}/T2M_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=3)
            save_path = f"{file_path}/T850_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=32)
            save_path = f"{file_path}/U850_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=45)
            save_path = f"{file_path}/Z500_{epoch}.png"
            utils.plot_fields(input_print, output_print-input_print, target_print-input_print, save_path=save_path, index=9)
        

def get_validation_mse(config_dict, model, dataloader, device, rank, epoch, file_path, mse_file_path='rmse.csv', acc_file_path='acc.csv', model_parallel=True, world_size=4, local_groups=None, node_groups=None, global_groups=None):
    if config_dict['model']['model_parallel']:
        mse_file_path = f'rmse_{rank}.csv'
        acc_file_path = f'acc_{rank}.csv'
    loss = torch.zeros(1).to(device)
    parallelism = config_dict['model_parallel']['parallelism']
    rollout_time = config_dict['model']['latent_rollout']
    dt = config_dict['data']['dt']
    rollout_time = (np.arange(rollout_time) + 1) * dt
    var_names = get_var_names(prepend_variable=['epoch', 'rollout_time'])
    n_lat, n_lon = config_dict['data']['xlat'], config_dict['data']['xlon']

    rmse, acc = {}, {}
    
    for rollout in rollout_time:
        rmse[rollout] = {}
        acc[rollout] = {}
        rmse[rollout]['epoch'] = epoch
        acc[rollout]['epoch']  = epoch
        
        rmse[rollout]['rollout_time'] = rollout
        acc[rollout]['rollout_time']  = rollout
        
        # each df has [epoch, rollout_time] as first two values
        rmse[rollout] = torch.zeros(len(var_names), device=device)
        acc[rollout]  = torch.zeros(len(var_names), device=device)
            
            
    total_samples = torch.tensor([0]).to(device)
    start_validation_time = time.perf_counter()

    surface_weights  = torch.tensor([1.5, 0.77, 0.77, 3.0]) * 4
    # [z, q, t, u, v]
    pressure_weights = torch.tensor([3.00, 0.6, 1.5, 0.9, 0.9]).unsqueeze(1).expand(5, 13).flatten()
    surface_upper_pressure_weights = torch.concat(
        [torch.tensor([1, 1, 1, 1]), # surface variables
        torch.tensor([1, 1, 1, 1, 1, 1, 0.9, 0.8, .7, .6, .5, .4, .3]).repeat(5)]
    ).view(1, 69)

    variable_weights = ((torch.concat([surface_weights, pressure_weights]).view(1, 1, 1, 69))*surface_upper_pressure_weights).to(device)
    
    # underground mask
    area_weights = torch.tensor(calc_weight(n_lat)).view(1, n_lat, 1, 1).to(device)
    
    # Load climatology
    if parallelism < 4:
        climatology = get_climatology()[:n_lat]
    elif parallelism == 4:
        n_lon = n_lon//2
        mod = rank % 4
        if mod in [0, 1]:
            climatology = get_climatology()[:n_lat, :n_lon]
        else:
            climatology = get_climatology()[:n_lat, n_lon:]

    if parallelism == 1:
        variable_weights = variable_weights.to(device)
    elif parallelism == 2:
        variable_weights = torch.nn.functional.pad(variable_weights, (0,3))
        variable_weights = variable_weights[0,0,0,:36] if rank % parallelism == 0 else variable_weights[0,0,0,36:]
    else:
        mod = rank % parallelism
        variable_weights = torch.nn.functional.pad(variable_weights, (0,3))
        variable_weights = variable_weights[0,0,0,:36] if rank % parallelism in [0, 2] else variable_weights[0,0,0,36:]

    xlat, xlon = config_dict['data']['xlat'], config_dict['data']['xlon']

    index_offset = 2 if 'rollout_time' in var_names else 1
    mean, std = load_averages()
    
    loss_fn = torch.nn.MSELoss()
    parallelism = config_dict['model_parallel']['parallelism']
    if parallelism == 4:
        buffer  = torch.zeros(config_dict['training']['valid_batch'], xlat, xlon//2, 36).to(device)
        buffer2 = torch.zeros(config_dict['training']['valid_batch'], xlat, xlon//2, 36).to(device)
    else:
        buffer, buffer2 = None, None
    
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Specify indices to calculate RMSE
    indices = np.array([0, 1, 2, 3, 6, 9, 19, 22, 32, 35, 45, 48, 58, 61])
    indices_key = indices + index_offset

    model.eval()
    loss = torch.tensor([0]).to(device)

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            for step, rollout in enumerate(rollout_time):

                # Write epoch 
                rmse[rollout][0] = epoch
                acc[rollout][0]  = epoch
                rmse[rollout][1] = step
                acc[rollout][1]  = step
        
                # Input and target data loading
                if parallelism == 1:
                    input_data  = data[0].to(device, non_blocking=True)
                    target_data = data[1][step].to(device, non_blocking=True)
                else:
                    if config_dict['model_parallel']['parallelism'] == 4:
                        input_data_or_handle, async_input = utils.load_and_push_data(data[0], device, non_blocking=False)
                        target_data_or_handle, async_target = utils.load_and_push_data(data[1][step], device, non_blocking=False)
                        if async_input:
                            input_data_or_handle.wait()
                            input_data = buffer
                        else:
                            input_data = input_data_or_handle
                    else:
                        input_data  = data[0].to(device, non_blocking=False)

                output = model(input_data, rollout_=step+1)
                output = output.detach()

                if parallelism == 1:
                    input_data = input_data.permute(0, 2, 3, 1) # Permute input data for mse calculation
                    target_data = target_data.permute(0, 2, 3, 1)

                    if batch == 0: # PLOT BEFORE MODIFYING
                        gen_plots(config_dict, input_data, output, target_data, rank, file_path, epoch, model_parallel=model_parallel)
                    
                    # Calculate RMSE and ACC
                    rmse[rollout][indices_key] = rmse[rollout][indices_key] + calc_mse_variable(target_data, output, device, area_weights, std=std, indices=indices).to(torch.float32)
                    acc[rollout][indices_key]  = acc[rollout][indices_key] + calc_acc_variable(target_data, output, device, area_weights, mean=mean, std=std, climatology=climatology, indices=indices).to(torch.float32)
                    
                    output_loss = output * area_weights * variable_weights
                    target_loss = target_data * area_weights * variable_weights

                elif parallelism == 2:
                    output = output.reshape(output.shape[0], xlat, xlon, 36)
                    
                    if rank % parallelism == 0:
                        norm_offset = 0
                        indices = np.array([0, 1, 2, 3, 6, 9, 19, 22, 32, 35])
                        target_data = data[1][step].to(device, non_blocking=False)
                    elif rank % parallelism == 1:
                        norm_offset = 36
                        indices = np.array([45, 48, 58, 61]) - norm_offset
                        target_data = data[1][step].to(device, non_blocking=False)
                    
                    
                    rmse[rollout][indices + norm_offset + index_offset] = rmse[rollout][indices + norm_offset + index_offset] + calc_mse_variable(target_data, output, device, area_weights, std=std, indices=indices, norm_offset=norm_offset).to(torch.float32)
                    acc[rollout][indices + norm_offset + index_offset]  = acc[rollout][indices + norm_offset + index_offset]  + calc_acc_variable(target_data, output, device, area_weights, mean=mean, std=std, climatology=climatology, indices=indices, norm_offset=norm_offset).to(torch.float32)
                    
                    if batch == 0: # PLOT BEFORE FURTHER MODIFICATIONS
                        gen_plots(config_dict, input_data, output, target_data, rank, file_path, epoch, model_parallel=model_parallel)
                    
                    # loss calculation
                    target_loss = target_data * area_weights * variable_weights
                    output_loss = output * area_weights * variable_weights
                    

                elif parallelism == 4:
                    output = output.reshape(output.shape[0], xlat, xlon//2, 36)               
                    # Wait for target data to be pushed if necessary
                    if async_target:
                        target_data_or_handle.wait()
                        target_data = buffer2
                    else:
                        target_data = target_data_or_handle

                    if rank % parallelism in [0,2]: # 0 and 2 handle the first half of the output variables
                        norm_offset = 0
                        indices = np.array([0, 1, 2, 3, 6, 9, 19, 22, 32, 35])
                    elif rank % parallelism in [1, 3]: # 1 and 3 handle the second half of the output variables
                        norm_offset = 36
                        indices = np.array([45, 48, 58, 61]) - norm_offset

                    # Each process contains its half of the squarred error
                    rmse[rollout][indices + norm_offset +  index_offset] = rmse[rollout][indices + norm_offset + index_offset] + calc_mse_variable(target_data, output, device, area_weights, std=std, indices=indices, norm_offset=norm_offset).to(torch.float32)
                    acc[rollout][indices + norm_offset + index_offset]   = acc[rollout][indices + norm_offset + index_offset]  + calc_acc_variable_parallel(target_data, output, device, area_weights, mean=mean, std=std, climatology=climatology, indices=indices, norm_offset=norm_offset, group=local_groups, rank=rank).to(torch.float32)

                    if batch == 0:
                        if  rank < 4 and parallelism == 4: # only needs to happen once
                            # Gather input data
                            # input data has shape [nvar, lat, lon]
                            input_data, output, target_data = input_data[0].contiguous(), output[0].contiguous(), target_data[0].contiguous()
                            
                            input_tensor_gather = torch.zeros(4, *input_data.shape).to(device)
                            torch.distributed.all_gather_into_tensor(input_tensor_gather, input_data, group=node_groups)
                            
                            top_row = torch.cat([input_tensor_gather[0], input_tensor_gather[1]], dim=2)
                            bottom_row = torch.cat([input_tensor_gather[2], input_tensor_gather[3]], dim=2)
                            
                            # left/right side shape: [1, 720, 720, 72]
                            input_data_gathered = torch.cat([top_row, bottom_row], dim=1)[:,:,:69]
                            
                            # gather first sample to rank 0
                            output_tensor_gather = torch.zeros(4, *output.shape).to(device)
                            
                            torch.distributed.all_gather_into_tensor(output_tensor_gather, output, group=node_groups)
                            
                            top_row2 = torch.cat([output_tensor_gather[0], output_tensor_gather[1]], dim=2)
                            bottom_row2 = torch.cat([output_tensor_gather[2], output_tensor_gather[3]], dim=2)
                            # left/right side shape: [1, 720, 720, 72]
                            output_gathered = torch.cat([top_row2, bottom_row2], dim=1)[:,:,:69] 
                            
                            # Gather target data. Currently split between 4 devices as well.
                            target_tensor_gather = torch.zeros(4, *target_data.shape).to(device)
                            target_data = target_data.contiguous()
                            torch.distributed.all_gather_into_tensor(target_tensor_gather, target_data, group=node_groups)
                            
                            top_row = torch.cat([target_tensor_gather[0], target_tensor_gather[1]], dim=2)
                            bottom_row = torch.cat([target_tensor_gather[2], target_tensor_gather[3]], dim=2)
                            # left/right side shape: [1, 720, 720, 72]
                            target_data_gathered = torch.cat([top_row, bottom_row], dim=1)[:,:,:69]
                            
                            if rank == 0:
                                gen_plots(config_dict, input_data_gathered, output_gathered, target_data_gathered, rank, file_path, epoch, model_parallel=model_parallel)    
                        
                    output_loss = output * area_weights * variable_weights
                    target_loss = target_data * area_weights * variable_weights
                    
                iter_loss = loss_fn(output_loss, target_loss)
                loss = loss + iter_loss.detach()

            total_samples = total_samples + input_data.shape[0]
            
    if parallelism == 1:
        torch.distributed.all_reduce(total_samples)
        torch.distributed.all_reduce(loss)
        loss /= total_samples
    else:
        total_samples = [len(dataloader.dataset)]
        loss = loss / world_size / total_samples[0] 
        
        torch.distributed.all_reduce(loss)    
    
    for rollout in rollout_time:
        if parallelism == 1:
            torch.distributed.all_reduce(rmse[rollout][2:]) 
            torch.distributed.all_reduce(acc[rollout][2:]) 
        elif parallelism == 2:
            # Each process has squared error of a certain variable. Need to add over the appropriate DDP groups = global groups.
            torch.distributed.all_reduce(rmse[rollout][2:], group=global_groups[rank%2]) 
            torch.distributed.all_reduce(acc[rollout][2:], group=global_groups[rank%2])  
        elif parallelism == 4:
            # First reduce over local groups --> [0,2], [1,3], [4,6]...
            torch.distributed.all_reduce(rmse[rollout][2:], group=local_groups[rank%2 + rank // 4 * 2]) # Adding MSE of two "partner" ranks together
            torch.distributed.all_reduce(rmse[rollout][2:], group=global_groups[rank%4]) # Then reduce over global ddp groups
            torch.distributed.all_reduce(acc[rollout][2:], group=global_groups[rank%4])
                
    for rollout in rollout_time:
        rmse[rollout][2:] = torch.sqrt(rmse[rollout][2:]/total_samples[0])
        acc[rollout][2:]  = acc[rollout][2:] / total_samples[0]

        if rank == 0:
            rmse[rollout][21:25] = rmse[rollout][21:25]*1000

    end_validation_time = time.perf_counter()

    if rank == 0:
        print(f"Validation time [min]: {(end_validation_time - start_validation_time)/60}, validation loss {loss.item()}")
        
    # write to file
    if (parallelism > 1 and rank < 2) or (parallelism == 1 and rank == 0):
        rmse_file, acc_file = Path(mse_file_path), Path(acc_file_path)
        for rollout in rollout_time:
            rmse_values = dict(zip(var_names, rmse[rollout].cpu().tolist()))
            acc_values  = dict(zip(var_names, acc[rollout].cpu().tolist()))
            
            df_rmse, df_acc = pd.DataFrame([rmse_values]), pd.DataFrame([acc_values])
            
            df_rmse.to_csv(mse_file_path, mode='a', header=False) if rmse_file.is_file() else df_rmse.to_csv(mse_file_path)
            df_acc.to_csv(acc_file_path, mode='a', header=False) if acc_file.is_file() else df_acc.to_csv(acc_file_path)

    return loss.item()

def load_averages():
    static_slevel = np.load('data/surface_zarr.npy')
    static_plevel = np.load('data/pressure_zarr.npy')
    mean_slevel   = torch.tensor(static_slevel[0].reshape(4))
    std_slevel    = torch.tensor(static_slevel[1].reshape(4))
    mean_plevel   = torch.tensor(static_plevel[0].reshape(65))
    std_plevel    = torch.tensor(static_plevel[1].reshape(65))

    mean = torch.concatenate([mean_slevel, mean_plevel], axis=0)
    std  = torch.concatenate([std_slevel, std_plevel], axis=0)
    return mean, std
    
def get_climatology():
    climatology_plevel = torch.tensor(np.load('/pressure_climatology.npy')).reshape(65, 721, 1440)
    climatology_slevel = torch.tensor(np.load('/surface_climatology.npy'))

    climatology = torch.concatenate([climatology_slevel, climatology_plevel], axis=0).permute(1, 2, 0)
    # climatology [721, 1440, variables]
    return climatology


def calc_mse_variable(target, result, device, area_weights, std, indices, norm_offset=0):
    n_lon = 1440
    n_lat = area_weights.numel()

    # target shape [n_batch, 721, 1440, vars]
    divisor = torch.sqrt(torch.tensor([n_lon*n_lat])).to(device)
    
    std_multiplier = std[indices+norm_offset].view(1, 1, 1, len(indices)).to(device)
    area_weights = area_weights.view(1, n_lat, 1, 1)
    
    mean_squared_error = ((target[:, :,:,indices] - result[:, :,:,indices]) * 
            std_multiplier.flatten() / divisor 
        )**2 * area_weights
    
    
    # MSE will have shape [n_batch, n_vars, ]
    mean_squared_error = mean_squared_error.sum(dim=(0,1,2)) 

    return mean_squared_error


def calc_acc_variable(target, result, device, weights, mean, std, climatology, indices=0, norm_offset=0):
    
    mean, std = mean[indices+norm_offset].to(device), std[indices+norm_offset].flatten().to(device)

    # climatology has shape [var, 721,1440] mean has shape    
    climatology = climatology[:,:,indices+norm_offset].to(device)
    climatology = (climatology - mean.view(1, 1, len(indices))).unsqueeze(0)
        
    std_multiplier = std.view(1, 1, 1, len(indices)).to(device)
    
    target_anomoly = (target[:, :, :, indices] * std_multiplier) - climatology
    result_anomoly = (result[:, :, :, indices] * std_multiplier) - climatology

    weights = weights.view(1, weights.numel(), 1, 1)
    # numerator has shape [batch_size, vars, n_lat, n_lon]
    numerator = torch.sum(weights * torch.mul(target_anomoly, result_anomoly), dim=(1,2)) 
    denom1    = torch.sum(weights * target_anomoly**2, dim=(1,2))
    denom2    = torch.sum(weights * result_anomoly**2, dim=(1,2))
    acc_vec   = numerator/(torch.sqrt(torch.mul(denom1, denom2)))
    
    acc_sum   = torch.sum(acc_vec, dim=(0))
    
    return acc_sum # should be a vector of length (indices)


def calc_acc_variable_parallel(target, result, device, weights, mean, std, climatology, indices=0, norm_offset=0, group=None, rank=0):
    # per process: target has shape [n_patches, nlat, nlon//2, nvar/2]
    
    mean, std = mean[indices+norm_offset].to(device), std[indices+norm_offset].flatten().to(device)
    
    climatology = climatology[:,:,indices+norm_offset].to(device)
    climatology = (climatology - mean.view(1, 1, len(indices))).unsqueeze(0)
    std_multiplier = std.view(1, 1, 1, len(indices)).to(device)
    
    target_anomoly = (target[:, :, :, indices] * std_multiplier) - climatology
    result_anomoly = (result[:, :, :, indices] * std_multiplier) - climatology

    weights = weights.view(1, weights.numel(), 1, 1)
        
    numerator = torch.sum(weights * torch.mul(target_anomoly, result_anomoly), dim=(1,2))
    denom1    = torch.sum(weights * target_anomoly**2, dim=(1,2))
    denom2    = torch.sum(weights * result_anomoly**2, dim=(1,2))
    
    torch.distributed.all_reduce(numerator, group=group[rank % 2 + rank // 4 * 2])
    torch.distributed.all_reduce(denom1, group=group[rank % 2 + rank // 4 * 2])
    torch.distributed.all_reduce(denom2, group=group[rank % 2 + rank // 4 * 2])

    acc_vec   = numerator/(torch.sqrt(torch.mul(denom1, denom2)))
    acc_sum   = torch.sum(acc_vec, dim=0)

    return acc_sum