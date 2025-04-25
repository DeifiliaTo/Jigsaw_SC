import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


class ERA5Dataset(Dataset):
    """Define 3D dataset."""

    def __init__(self, config, file_path, mode, forecast_length=1, normalize=True, rank=None):
        """
        Initialize.

        params: Dict
            configuration file
        file_path: String
            path to data directory
        distributed: bool
            flag for DDP
        mode: String
            of value 'training', 'testing', 'validation'
        device: String
            device that the code is running/offloaded on
        forecast_length: int
            For training, always 1. For validation, defines the number of autoregressive steps to roll out to.
        """
        self.file_path = file_path
        self.mode = mode
        self.dt = config['data']['dt']
        self.deltaTDivisor = 6
        self.forecast_length = forecast_length
        self.normalize = normalize
        
        self._get_files_stats(mode=self.mode)
        
        # If data is downloaded from weatherbench, the 6-hourly subsampled data is stored in the reverse pressure level order.
        # Need to reverse so that the 0th presssure level is 1000 hPa, 1st is 925 hPa, etc.
        self.level_ordering = range(13-1, -1, -1)

        self.p_mean = np.load(f'{os.getenv("SRC")}/data/pressure_zarr.npy', allow_pickle=True)[0].reshape(5, 13, 1, 1)
        self.p_std  = np.load(f'{os.getenv("SRC")}/data/pressure_zarr.npy', allow_pickle=True)[1].reshape(5, 13, 1, 1)
        self.s_mean = np.load(f'{os.getenv("SRC")}/data/surface_zarr.npy', allow_pickle=True)[0].reshape(4, 1, 1)
        self.s_std  = np.load(f'{os.getenv("SRC")}/data/surface_zarr.npy', allow_pickle=True)[1].reshape(4, 1, 1)
        
        self.xlat = config['data']['xlat']
        self.rank = rank        
        
        
    def _get_files_stats(self, mode='train'):
        """Filter desired time points based on parameters and return file statistics."""
        self.zarr_data = xr.open_dataset(self.file_path, engine='zarr')
        times = pd.to_datetime(self.zarr_data['time'].values)
        if mode == 'train': # training case, lite
            train_years = times[(times.year<2018) & (times.year > 1979) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == 'validation':           # validation
            validation_years = times[(times.year == 2018) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
            
        self.n_samples_total = len(self.zarr_data['time'])
        
        
    def __len__(self):
        """Return total number of samples."""
        return self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor  - 1 # -1 to avoid last data point
    
    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        skipped_ranks = [1, 2, 3]    
        if self.rank is not None:
            if self.rank in skipped_ranks: # 0, 1 always senders
                return [torch.zeros(1), torch.zeros(1)], [torch.zeros(1), torch.zeros(1)]
        
        target_pressure = []
        target_surface  = []
        step = self.dt 

        # Get target file indices
        target_file_idxs = global_idx + torch.div(torch.arange(1, self.forecast_length+1)*step, self.deltaTDivisor , rounding_mode='trunc')
            
        for target_idx in target_file_idxs:
            if target_idx >= self.__len__(): 
                target_idx = self.__len__() - 1
            
        # level flag ensures the data is read with the correct pressure level ordering
        input_pressure = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        input_surface  = self.zarr_data.isel(time=global_idx)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
        target_pressure_ds = self.zarr_data.isel(time=target_file_idxs, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        target_surface_ds  = self.zarr_data.isel(time=target_file_idxs)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
        # Stack and convert to numpy array
        input_pressure = np.stack([input_pressure['geopotential'].values, input_pressure['specific_humidity'].values, input_pressure['temperature'].values, input_pressure['u_component_of_wind'].values, input_pressure['v_component_of_wind'].values], axis=0)
        input_surface  = np.stack([input_surface['mean_sea_level_pressure'].values, input_surface['10m_u_component_of_wind'].values, input_surface['10m_v_component_of_wind'].values, input_surface['2m_temperature'].values], axis=0)
        
        # Normalize
        input_pressure = (input_pressure - self.p_mean) / self.p_std
        input_surface  = (input_surface  - self.s_mean) / self.s_std
        input_pressure = np.reshape(input_pressure, (-1, 721, 1440))
        # Perform necessary cropping
        input_surface  = input_surface[:, :self.xlat]
        input_pressure = input_pressure[:, :self.xlat]
        
        input = np.concatenate([input_surface, input_pressure], axis=0)
        
        target = []
        for i in range(self.forecast_length):
            pressure_ds = target_pressure_ds.isel(time=i)
            surface_ds  = target_surface_ds.isel(time=i)
            target_pressure.append(np.stack([pressure_ds['geopotential'].values, pressure_ds['specific_humidity'].values, pressure_ds['temperature'].values, pressure_ds['u_component_of_wind'].values, pressure_ds['v_component_of_wind'].values], axis=0))
            target_surface.append(np.stack([surface_ds['mean_sea_level_pressure'].values, surface_ds['10m_u_component_of_wind'].values, surface_ds['10m_v_component_of_wind'].values, surface_ds['2m_temperature'].values], axis=0))
            target_pressure[i] = (target_pressure[i] - self.p_mean) / self.p_std
            target_surface[i]  = (target_surface[i] - self.s_mean) / self.s_std
            target_pressure[i], target_surface[i] = torch.tensor(target_pressure[i]), torch.tensor(target_surface[i])
            
            target_pressure[i] = target_pressure[i].reshape(-1, 721, 1440)
            target_pressure[i] = target_pressure[i][:, :self.xlat]
            target_surface[i] = target_surface[i][:, :self.xlat]
            target.append(torch.concat([target_surface[i], target_pressure[i]], axis=0))

        input = torch.tensor(input)
        
        return input, target    
        
class ERA5DatasetDistributed(Dataset):
    """Define 2D dataset."""

    def __init__(self, config, file_path, mode, rank, forecast_length=1, normalize=True, dt=None):
        """
        Initialize.

        config: Dict
            configuration file
        file_path: String
            path to data directory
        mode: String
            of value 'training', 'testing', 'validation'
        rank: int
            rank of process
        forecast_length: int
            Defines the number of forecast time steps to return. Used in validation or in fine-tuning on longer rollouts
        normalize: Boolean
            flag for whether the input and target data should be normalized. Defaults to True.
        dt: int
            forecast lead time. Defaults to value in config file, unless overridden.
        """
        self.file_path = file_path
        self.mode = mode
        self.dt = dt if dt is not None else config['data']['dt']
        self.rank = rank
        self.datapath = config['data']['train_data_path']     

        self.deltaTDivisor = 6
        self.forecast_length = forecast_length
        self.normalize = normalize
        
        self._get_files_stats(mode=self.mode)
        
        # If data is downloaded from weatherbench, the 6-hourly subsampled data is stored in the reverse pressure level order.
        # Need to reverse so that the 0th presssure level is 1000 hPa, 1st is 925 hPa, etc.
        if config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/1959-2023_01_10-wb13-6h-1440x721.zarr':
            self.level_ordering = range(13-1, -1, -1)
        elif config['data']['train_data_path'] == '/lsdf/kit/imk-tro/projects/Gruppe_Quinting/ec.era5/era5.zarr':
            self.level_ordering = range(0, 13)
        elif config['data']['train_data_path'] == '/hkfs/work/workspace/scratch/ke4365-era5_data/era5.zarr':
            self.level_ordering = range(0, 13)
        else: # baseline assuption is subsampled WB2 data
            self.level_ordering = range(13-1, -1, -1)

        # normalization for pressure levels and standard deviation
        self.p_mean = np.load(f'{os.getenv("SRC")}/data/pressure_zarr.npy', allow_pickle=True)[0].reshape(65, 1, 1)
        self.p_std  = np.load(f'{os.getenv("SRC")}/data/pressure_zarr.npy', allow_pickle=True)[1].reshape(65, 1, 1)
        self.s_mean = np.load(f'{os.getenv("SRC")}/data/surface_zarr.npy', allow_pickle=True)[0].reshape(4, 1, 1)
        self.s_std  = np.load(f'{os.getenv("SRC")}/data/surface_zarr.npy', allow_pickle=True)[1].reshape(4, 1, 1)
        
        self.xlat, self.xlon = config['data']['xlat'], config['data']['xlon']
        self.batch_size = config['training']['train_batch']
        self.split = config['model_parallel']['split']
        self.parallelism = config['model_parallel']['parallelism']

        if self.parallelism == 4:
            if self.split == 4:
                if self.rank % 4 in [0, 1]:
                    self.start, self.end = 0, self.xlon//2
                else:
                    self.start, self.end = self.xlon//2, self.xlon
            else: 
                self.start, self.end = 0, self.xlon
        else:
            self.start, self.end = 0, self.xlon
               
        self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(xlat=self.xlat)
        self.land_mask = torch.unsqueeze(self.land_mask[:,self.start:self.end], 0)
        self.soil_type = torch.unsqueeze(self.soil_type[:,self.start:self.end], 0)
        self.topography = torch.unsqueeze(self.topography[:,self.start:self.end], 0)

    def _get_files_stats(self, mode='train'):
        """Filter desired time points based on parameters and return file statistics."""
        self.zarr_data = xr.open_dataset(self.file_path, engine='zarr')
        times = pd.to_datetime(self.zarr_data['time'].values)
        if mode == 'train': # training case, lite
            train_years = times[(times.year<2017) & (times.year > 1979) & ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            self.zarr_data = self.zarr_data.sel(time=train_years)
        elif mode == 'validation':           # validation
            validation_years = times[(times.year == 2018)& ((times.hour == 0) | (times.hour == 6) | (times.hour == 12) | (times.hour == 18))]
            self.zarr_data = self.zarr_data.sel(time=validation_years)
        
        self.n_samples_total = len(self.zarr_data['time'])
        
        # N channels
        self.n_in_channels   = len(self.zarr_data.data_vars)
        
        
    def __len__(self):
        """Return total number of samples."""
        return self.n_samples_total - self.forecast_length * self.dt // self.deltaTDivisor  
    
        
    def __getitem__(self, global_idx):
        """
        Return single input, target.

        global_idx: int
                global index of item
        """
        target_surface  = []
        step = self.dt
        # if forecast_length = 3, target_file_idx = global_idx + range(1, forecast_length+1)*step // 6
        target_file_idxs = global_idx + torch.div(torch.arange(1, self.forecast_length+1)*step, self.deltaTDivisor , rounding_mode='trunc')
        for target_idx in target_file_idxs:
            if target_idx >= self.__len__(): 
                target_idx = self.__len__() - 1
            
        # Isolate data from time point and convert to numpy array
        # WeatherBench data stores from low --> high pressure levels
        # We convert to high --> low
        input_pressure = self.zarr_data.isel(time=global_idx, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        input_surface  = self.zarr_data.isel(time=global_idx)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
        target_pressure_ds = self.zarr_data.isel(time=target_file_idxs, level=self.level_ordering)[['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']]
        target_surface_ds  = self.zarr_data.isel(time=target_file_idxs)[['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
            
        target = []
        if self.rank % 2 == 0: # first half of data
            # Input data
            # Stack and convert to numpy array
            geopotential = input_pressure['geopotential'][:,:self.xlat,self.start:self.end].values
            humidity = input_pressure['specific_humidity'][:,:self.xlat,self.start:self.end].values
            temperature = input_pressure['temperature'][:6,:self.xlat,self.start:self.end].values
            
            # Stack relevant surface variables
            mslp = input_surface['mean_sea_level_pressure'][:self.xlat,self.start:self.end].values
            u10  = input_surface['10m_u_component_of_wind'][:self.xlat,self.start:self.end].values
            v10  = input_surface['10m_v_component_of_wind'][:self.xlat,self.start:self.end].values
            t2m  = input_surface['2m_temperature'][:self.xlat,self.start:self.end].values
            
            # Concatenate surface and pressure variables
            input_pressure = np.concatenate([geopotential, humidity, temperature], axis=0)
            input_surface  = np.stack([mslp, u10, v10, t2m], axis=0)
            # Normalize
            input_pressure = (input_pressure - self.p_mean[:32]) / self.p_std[:32]
            input_surface  = (input_surface  - self.s_mean) / self.s_std
            input_pressure = np.reshape(input_pressure, (-1, self.xlat, self.end-self.start))
            
            input = np.concatenate([input_surface, input_pressure], axis=0)
            
            # Target data
            for i in range(self.forecast_length):
                    
                pressure_ds = target_pressure_ds.isel(time=i)
                surface_ds  = target_surface_ds.isel(time=i)
                
                geopotential = pressure_ds['geopotential'][:,:self.xlat,self.start:self.end].values
                humidity = pressure_ds['specific_humidity'][:,:self.xlat,self.start:self.end].values
                temperature = pressure_ds['temperature'][:6,:self.xlat,self.start:self.end].values
                
                mslp = surface_ds['mean_sea_level_pressure'][:self.xlat,self.start:self.end].values
                u10  = surface_ds['10m_u_component_of_wind'][:self.xlat,self.start:self.end].values
                v10  = surface_ds['10m_v_component_of_wind'][:self.xlat,self.start:self.end].values
                t2m  = surface_ds['2m_temperature'][:self.xlat,self.start:self.end].values
                temp = np.concatenate([geopotential, humidity, temperature], axis=0)
                target_surface.append(np.stack([mslp, u10, v10, t2m], axis=0))

                temp = (temp - self.p_mean[:32]) / self.p_std[:32]
                target_surface[i]  = (target_surface[i] - self.s_mean) / self.s_std
                temp = temp.reshape(-1, self.xlat, self.end-self.start)
                
                temp = np.concatenate([target_surface[i], temp], axis=0)
                if self.split == 2 and self.parallelism == 4:
                    target.append([torch.tensor(temp[:,:,:720]).permute(1, 2, 0), torch.tensor(temp[:,:,720:]).permute(1, 2, 0)])
                else:
                    target.append(torch.tensor(temp).permute(1, 2, 0))
        else:
            temperature = input_pressure['temperature'][6:,:self.xlat,self.start:self.end].values
            u_velocity  = input_pressure['u_component_of_wind'][:,:self.xlat,self.start:self.end].values
            v_velocity = input_pressure['v_component_of_wind'][:,:self.xlat,self.start:self.end].values
            
            input = np.concatenate([temperature, u_velocity, v_velocity], axis=0)
            
            input = (input - self.p_mean[32:]) / self.p_std[32:]
            input = np.reshape(input, (-1, self.xlat, self.end-self.start))
            input = np.concatenate([input, self.land_mask, self.soil_type, self.topography], axis=0)

            for i in range(self.forecast_length):
                pressure_ds = target_pressure_ds.isel(time=i)
                temperature = pressure_ds['temperature'][6:,:self.xlat,self.start:self.end].values
                u_velocity  = pressure_ds['u_component_of_wind'][:,:self.xlat,self.start:self.end].values
                v_velocity = pressure_ds['v_component_of_wind'][:,:self.xlat,self.start:self.end].values
                
                temp = np.concatenate([temperature, u_velocity, v_velocity], axis=0)
                
                temp = (temp - self.p_mean[32:]) / self.p_std[32:]
                
                temp = temp.reshape(-1, self.xlat, self.end-self.start) 
                
                target.append(torch.nn.functional.pad(torch.tensor(temp), (0, 0, 0, 0, 0, 3)).permute(1, 2, 0)) 
                            
        input = torch.tensor(input).permute(1,2,0)
        
        return input, target



def load_constant_mask_2d(
    folder_path="/hkfs/work/workspace/scratch/ke4365-summa/constant_masks/",
    xlat=721
):
    """
    Load the constant masks applied in the patch embedding layer.

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
