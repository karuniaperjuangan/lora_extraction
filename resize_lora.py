import argparse
import os
from huggingface_hub import list_repo_files, snapshot_download
cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/transformers"))

import safetensors
import safetensors.torch
import json
import pickle
import torch
import glob
import os
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft.tuners.lora import QuantLinear
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig
from typing import Union
import shutil
parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--lora_rank", type=int, required=True)




def load_safetensors(directory, framework="pt", device='cpu'):
    """
    Loads tensors from .safetensors files in the specified directory into a dictionary.

    Args:
    - directory (str): Path to the directory containing .safetensors files.
    - framework (str): The framework to use ('pt' for PyTorch, 'tf' for TensorFlow, etc.). Default is 'pt'.
    - device (str): The device to load the tensors on ('cpu', 'cuda', etc.). Default is 'cpu'.

    Returns:
    - dict: A dictionary containing all tensors from the .safetensors files.
    """
    tensors_dict = {}
    # Use glob to find all .safetensors files in the directory
    file_paths = glob.glob(f"{directory}/*.safetensors")

    # Loop through each file and load its tensors into the dictionary
    for file_path in sorted(file_paths):
        with safetensors.safe_open(file_path, framework=framework, device=device) as f:
            for k in f.keys():
                tensors_dict[k] = f.get_tensor(k)

    return tensors_dict



def reduce_rank(weight, type:str,reduced_rank=16):

    if weight.dim() != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {weight.dim()} dimensions.")

    if type == "A":
        weight = weight[:reduced_rank, :].contiguous()
    elif type == "B":
        weight = weight[:, :reduced_rank].contiguous()
    else:
        raise ValueError(f"Unknown type {type}")

    return weight



def get_module_peft_name(module_name):
    return module_name.split('.')[-1]

if __name__ == "__main__":
    args = parser.parse_args()
    
    with open(os.path.join(args.input_dir, 'adapter_config.json')) as f:
        config = json.load(f)
    if config['r'] < args.lora_rank:
        raise ValueError(f"LoRA rank {args.lora_rank} must be smaller than the rank used for the adapter {config['r']}")

    adapter_dict = load_safetensors(args.input_dir)

    new_adapter_dict = {}
    for k, v in adapter_dict.items():
        
        if 'lora_A' in k:
            print(f"Reducing rank for {k}")
            new_adapter_dict[k] = reduce_rank(v, 'A', args.lora_rank)
        elif 'lora_B' in k:
            print(f"Reducing rank for {k}")
            new_adapter_dict[k] = reduce_rank(v, 'B', args.lora_rank)
        else:
            print(f"Skipping {k}")
            new_adapter_dict[k] = v
        print(f"Old {k} :", adapter_dict[k].shape)
        print(f"New {k} :", new_adapter_dict[k].shape)

    os.makedirs(args.output_dir, exist_ok=True)
    config['r'] = args.lora_rank
    print(f"Saving new adapter config to {args.output_dir}")
    with open(os.path.join(args.output_dir, 'adapter_config.json'), 'w') as f:
        json.dump(config, f,indent=2)
    print(f"Saving new adapter model to {args.output_dir}")
    safetensors.torch.save_file(new_adapter_dict, os.path.join(args.output_dir, 'adapter_model.safetensors'))
    list_copied_files = [file for file in os.listdir(args.input_dir) if file not in ['adapter_config.json', 'adapter_model.safetensors']]
    for file in list_copied_files:
        print(f"Copying {file}")
        shutil.copy(os.path.join(args.input_dir, file), os.path.join(args.output_dir, file))
