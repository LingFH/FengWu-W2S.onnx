import os
import numpy as np
import onnx
import onnxruntime as ort
import importlib.util
from onnx.utils import DataScaler,generate_fractal_noise_2d
from typing import Dict, List, Tuple, Optional
import xarray as xr
from pathlib import Path
from tqdm import tqdm


class FengWu_W2S_128:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()
        self.data_scaler = DataScaler()
        self.all_variables = [(vname, None) for vname in self.single_level_vnames] + \
                             [(vname, height) for vname in self.multi_level_vnames for height in self.height_level]      
        self.initialize_session()


    def load_config(self):
        """Load configuration from file and set default values if not specified."""
        try:
            spec = importlib.util.spec_from_loader("config", loader=None)
            config_module = importlib.util.module_from_spec(spec)
            with open(self.config_file, 'r') as f:
                exec(f.read(), config_module.__dict__)
            
            # Set default values if not specified in the config file
            self.model_path = getattr(config_module, 'model_path',  '../onnx/fengwu_w2s_128.onnx')
            self.session_options = ort.SessionOptions() #getattr(config_module, 'session_options', {})
            self.cuda_provider_options = getattr(config_module, 'cuda_provider_options', {})
            self.providers = getattr(config_module, 'providers', ['CPUExecutionProvider'])
            self.pred_step = getattr(config_module, 'pred_step', 10)
            self.ensembles_nums = getattr(config_module, 'ensembles_nums', 5)
            self.n_level = getattr(config_module, 'n_level', 0.1)
            self.device = getattr(config_module, 'device', 'cpu')
            self.save_nc = getattr(config_module, 'save_nc', True)
            self.output_path = getattr(config_module, 'output_path', './predictions')
            self.single_level_vnames = getattr(config_module, 'single_level_vnames', 
                                              ['u10', 'v10', 'msl', 'u100', 'v100', 't2m', 'sst', 'tp6h', 'swvl1', 'mtnlwrf', 'swh', 'mwd', 'mwp'])
            self.multi_level_vnames = getattr(config_module, 'multi_level_vnames', ['z', 'q', 'u', 'v', 't'])
            self.height_level = getattr(config_module, 'height_level', [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
            
                    
        except Exception as e:
            raise RuntimeError(f"Config loading failed: {str(e)}")
    
    def initialize_session(self):
        """Initialize the ONNX Runtime session with the current settings."""
        self.session_options.enable_cpu_mem_arena=False
        self.session_options.enable_mem_pattern = False
        self.session_options.enable_mem_reuse = False
        self.session_options.intra_op_num_threads = 1

        # ort.set_default_logger_severity(0)
        self.ort_session = ort.InferenceSession(
            self.model_path,
            sess_options=self.session_options,
            providers=[('CUDAExecutionProvider', self.cuda_provider_options)] if 'CUDAExecutionProvider' in self.providers else []
        )

    def prepare_input(self, input_data_step1, input_data_step2):
        """
        Prepare the input data to match the expected shape and type.
        the data order is single_level then z [50~1000] q ..... t
        :param input_data: The raw input data for inference.
        :return: A dictionary with the prepared input tensor.
        """
        assert input_data_step1.shape[1] == len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level)
        
        start_idx = 0
        normalized_data_1 = []
        normalized_data_2 = []
        for vname, height in self.all_variables:
            # print(vname)
            mean, std = self.data_scaler.got_meanstd(vname, height)
            # print(vname,mean,std)
            if vname == 'tp6h':
                # print(start_idx)
                temp_step1 = np.log(input_data_step1[:, start_idx:start_idx+1]+1)
                temp_step2 = np.log(input_data_step2[:, start_idx:start_idx+1]+1)
            elif vname in ["sst",'mwd','mwp','swh'] :
                climdict={'sst':286.8021042805777,'mwd':[188.9779577972233],"mwp":[8.569393484324436],"swh":[2.459809949498708]}
                temp_step1=np.where(np.isnan(input_data_step1[:, start_idx:start_idx+1]),climdict[vname],input_data_step1[:, start_idx:start_idx+1])
                temp_step2=np.where(np.isnan(input_data_step2[:, start_idx:start_idx+1]),climdict[vname],input_data_step2[:, start_idx:start_idx+1])
                temp_step1 = (temp_step1 - mean)/ std
                temp_step2 = (temp_step2 - mean)/ std
            else:
                temp_step1 = (input_data_step1[:, start_idx:start_idx+1] - mean)/ std
                temp_step2 = (input_data_step2[:, start_idx:start_idx+1] - mean)/ std
                # print(start_idx)
            normalized_data_1.append(temp_step1)
            normalized_data_2.append(temp_step2)
            start_idx += 1
        normalized_data_1 = np.concatenate(normalized_data_1, axis=1)
        normalized_data_2 = np.concatenate(normalized_data_2, axis=1)
        
        input = np.concatenate((normalized_data_1, normalized_data_2), axis=1)[:, :, :, :]

        return input.astype(np.float32)
    
    def run_inference(self, input_data):
        """
        Run inference using the provided input data.

        :param input_data: Prepared input data as a dictionary.
        :return: Output from the model.
        """

        input = input_data

        pred_list = []
        for i in tqdm(range(self.pred_step), desc="Prediction Steps"):
            ensemble_prediction = []
            for ensemble_num in range(self.ensembles_nums):
                if  i == 0 and ensemble_num != 0:

                    noise_shape = [input.shape[0]*input.shape[1], input.shape[2], input.shape[3]]
                    noise = generate_fractal_noise_2d(noise_shape, [4, 4], octaves=3, tileable=(False, True), device=self.device).cpu().numpy()
                    inp = input + self.n_level * noise.reshape(input.shape)
                elif i == 0 and ensemble_num == 0:
                    inp = input
                else:
                    inp = pred_list[ensemble_num]


                # print(inp.max())

                output = self.ort_session.run(None, {'input':inp})[0]


                if i == 0:
                    next_inp = np.concatenate((inp[:,inp.shape[1]//2:], output[:, :inp.shape[1]//2]), axis=1)
                    pred_list.append(next_inp)
                
                else:
                    pred_list[ensemble_num] = np.concatenate((pred_list[ensemble_num][:,inp.shape[1]//2:], output[:, :inp.shape[1]//2]), axis=1)

                
                
                denorm_output = self.denormalize_output(output[:, :inp.shape[1]//2])
                ensemble_prediction.append(denorm_output[:,None,:,:,:])
            
            ensemble_prediction = np.concatenate(ensemble_prediction, axis=1) # (batch,ensemble_number,variable,lat,lon)
            if self.save_nc :
                Path(self.output_path).mkdir(parents=True, exist_ok=True)
                nc_file_path = self.output_path + f"/Prediction_{(i+1)*6}h.nc"
                self._create_dataset(ensemble_prediction).to_netcdf(nc_file_path)
                print(f"Saved NetCDF file: {nc_file_path}")
        return print("done") 
    
    def _create_dataset(self, data) -> xr.Dataset:
        """
        :param data: [steps, member, channels, lat, lon]
        :return: xarray Dataset
        """
        variable_attributes = {
            'tp6h': {
                    'units': 'mm',
                    'long_name': 'Total Precipitation over 6 hours',
                    'standard_name': 'precipitation_amount'
                    },
        'other_var': {
                    'units': '',
                    'long_name': ' ',
                    'standard_name': ''
                    }
                }

        lat = np.linspace(90, -90, 128)
        lon = np.arange(0,360,1.40625)
        number = self.ensembles_nums
        time = np.arange(data.shape[0]) 

        data_vars = {}
        var_idx = 0
        

        for var in self.single_level_vnames:
            data_vars[var] = xr.DataArray(
                data[:, :, var_idx],
                dims=('time', 'member', 'lat', 'lon'),
                attrs=variable_attributes.get(var, {})
            )
            var_idx += 1


        for var in self.multi_level_vnames:
            data_vars[f"{var}"] = xr.DataArray(
                    data[:, :, var_idx:var_idx+len(self.height_level)],
                    dims=('time', 'member', 'level','lat', 'lon'),
                    attrs=variable_attributes.get(var, {})
                )
            var_idx += len(self.height_level)
                
        return xr.Dataset(
            data_vars,
            coords={
                'time': time,
                'member':np.arange(number),
                'level': self.height_level,
                'lat': lat,
                'lon': lon
            },
            attrs={'model': 'FengWu-W2S-140km',
                    "institution": "Shanghai Ailab",
                    "contact": "Fenghua Ling@lfhnuist@hotmail.com"}
        )


    def denormalize_output(self, output):
        """
        Denormalize the output data using the same normalization parameters.
        
        :param output: Raw output from the model.
        :return: Denormalized output data.
        """
        land_sea_mask = np.load("./onnx/Land_sea.npy")
        start_idx = 0
        denorm_output_list = []
        for vname, height in self.all_variables:
            mean, std = self.data_scaler.got_meanstd(vname, height)
            if vname == 'tp6h':
                denorm_output = (np.exp(output[:, start_idx:start_idx+1]) - 1)*1000
                denorm_output[denorm_output<0.1] = 0
            else:
                denorm_output = output[:, start_idx:start_idx+1] * std + mean
            if vname in ["sst",'mwd','mwp','swh'] :
                denorm_output = np.where(land_sea_mask==1,np.nan,denorm_output)
            denorm_output_list.append(denorm_output)
            start_idx += 1
        
        denorm_output = np.concatenate(denorm_output_list, axis=1)
        return denorm_output








