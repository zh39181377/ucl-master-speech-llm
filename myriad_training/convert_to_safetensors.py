import torch
from safetensors.torch import save_file
import os

model_path = "."
bin_file = os.path.join(model_path, "pytorch_model.bin")
safe_file = os.path.join(model_path, "model.safetensors")

print(f"Loading weights from {bin_file} (this takes a lot of RAM)...")

state_dict = torch.load(bin_file, map_location="cpu")

print(f"Saving weights to {safe_file} in safetensors format...")

save_file(state_dict, safe_file)

print("Conversion complete! You can now safely delete the original .bin file.")