# Configuration for FengWu_W2S_128 class

# Default values
single_level_vnames = ['u10', 'v10', 'msl', 'u100', 'v100', 't2m', 'sst', 'tp6h', 'swvl1', 'mtnlwrf', 'swh', 'mwd', 'mwp']
multi_level_vnames = ['z', 'q', 'u', 'v', 't']
height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# ONNX Runtime session options
session_options = {}

# CUDA provider options (if using GPU)
cuda_provider_options = {
    'device_id': 4,
    'arena_extend_strategy': 'kSameAsRequested',
    # 'intra_op_num_threads':1
}

# Providers list
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Other necessary parameters
pred_step = 168  # Number of prediction steps
ensembles_nums = 10  # Number of ensembles
n_level = 0.09  # Noise level for ensemble generation
device = 'cuda'  # Device to use ('cpu' or 'cuda')
save_nc = True  # Whether to save predictions as NetCDF files
output_path = './predictions'  # Path to save the output NetCDF files

# Model path
model_path = '/mnt/petrelfs/lingfenghua/S2S/FengWu-W2S.onnx/140km/FengWu_W2S_128.onnx'
