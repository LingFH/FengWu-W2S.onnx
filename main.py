from onnx.fengwu_w2s_128 import FengWu_W2S_128
import numpy as np
import os

# Define paths
config_file ='./configs/FengWu_W2S_setting.py'
# Create an instance of FengWu_W2S_128
print("Starting the inference process...")
fw_w2s = FengWu_W2S_128(config_file=config_file)

# Generate example input data
input_data_step1 = np.load("input_data0.npy")[np.newaxis,:,:,:]
input_data_step2 =np.load("input_data1.npy")[np.newaxis,:,:,:]

# Prepare input data
prepared_input = fw_w2s.prepare_input(input_data_step1, input_data_step2)

print("Input data prepared successfully.")
# print(prepared_input)
# Run inference
print("Running inference...")
fw_w2s.run_inference(prepared_input)