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

parser = argparse.ArgumentParser()

parser.add_argument("--base_model_name_or_path", required=True)
parser.add_argument("--finetuned_model_name_or_path", required=True)
parser.add_argument("--output_dir", required=False)
parser.add_argument("--lora_rank", type=int)
parser.add_argument("--lora_alpha", type=float, default=1)

def download_transformers_model(repo_id, cache_dir=cache_dir):
    # Check for .safetensors files in the repository
    repo_files = list_repo_files(repo_id)
    has_safetensors = any(file.endswith('.safetensors') for file in repo_files)

    # Define ignore_patterns based on the presence of .safetensors files
    ignore_patterns = ["*.bin"] if has_safetensors else None

    # Download the repository, ignoring PyTorch .bin files if .safetensors files are present
    local_path = snapshot_download(repo_id=repo_id,
                                    cache_dir=cache_dir,
                                    ignore_patterns=ignore_patterns,
                                    )

    print(f"Model downloaded to: {local_path}")
    if has_safetensors:
        print("Note: PyTorch .bin files were ignored due to the presence of .safetensors files.")
    return os.path.abspath(local_path), has_safetensors

def load_pytorch_tensors(directory, device='cpu'):
    """
    Loads tensors from .bin files in the specified directory into a dictionary.

    Args:
    - directory (str): Path to the directory containing .bin files.
    - device (str): The device to load the tensors on ('cpu', 'cuda', etc.). Default is 'cpu'.

    Returns:
    - dict: A dictionary containing all tensors from the .bin files.
    """
    tensors_dict = {}
    # Use glob to find all .bin files in the directory
    file_paths = glob.glob(f"{directory}/*.bin")

    # Loop through each file and load its tensors into the dictionary
    for file_path in sorted(file_paths):
        loaded_tensors = torch.load(file_path, map_location=torch.device(device))
        for k, v in loaded_tensors.items():
            tensors_dict[k] = v

    return tensors_dict

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

def get_linear_embedding_layers(model_type):
    """
    returns the linear embedding layers needed for loras, dependent on the model arch
    """
    if model_type == "gpt_neox":
        return ["embed_in", "embed_out"]
    if model_type == "falcon":
        return ["word_embeddings", "lm_head"]
    return ["embed_tokens", "lm_head"]


def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)

    names = []
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
            and name not in get_linear_embedding_layers(None)
        ):
            names.append(name)


    return names

def get_linear_module_names(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, state_dict={}, device_map="meta") #avoid loading weights as we won't need them
    return find_all_linear_names(model)

def _low_rank_decomposition(weight, reduced_rank=16):
    """
    Decompose a 2D matrix into low-rank matrices A and B using SVD.a

    :param weight: The matrix to decompose, of shape (H, W)
    :param reduced_rank: The final rank of the decomposition
    :return: A tuple of tensors (A, B)
    """
    if weight.dim() != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {weight.dim()} dimensions.")

    # SVD Decomposition
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

    # Truncated matrices
    A = Vt[:reduced_rank, :]
    B = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank])

    return A, B

def decompose_delta_weight(new_weight, base_weight, alpha, reduced_rank, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    new_weight = new_weight.to(device).float()
    base_weight = base_weight.to(device).float()
    

    """
    Decompose the delta weight into low-rank matrices A and B, considering the alpha scaling factor.

    :param new_weight: The updated weight matrix after applying LoRA.
    :param base_weight: The original weight matrix before LoRA.
    :param alpha: The alpha scaling factor used in LoRA.
    :param reduced_rank: The rank for the low-rank decomposition.
    :return: A tuple of tensors (A, B)
    """
    delta_weight = new_weight - base_weight

    # Check if alpha is applied uniformly
    # Adjust the implementation if alpha is applied differently
    adjusted_delta_weight = delta_weight / alpha

    A, B = _low_rank_decomposition(adjusted_delta_weight, reduced_rank=reduced_rank)

    return A, B

def get_module_peft_name(module_name):
    return module_name.split('.')[-1]

if __name__ == "__main__":
    args = parser.parse_args()
    
    base_model_download_path, base_model_has_safetensors = download_transformers_model(args.base_model_name_or_path)
    finetuned_model_download_path, finetuned_model_has_safetensors = download_transformers_model(args.finetuned_model_name_or_path)

    models = {
        'base' : {
            'download_path' : base_model_download_path,
            'has_safetensors' : base_model_has_safetensors
        },
        'finetuned' : {
            'download_path' : finetuned_model_download_path,
            'has_safetensors' : finetuned_model_has_safetensors
        }
    }

    linear_module_names = get_linear_module_names(models['base']['download_path'])

    try:
        linear_module_names.remove('lm_head')
    except:
        pass
    try:
        linear_module_names.remove('embed_tokens')
    except:
        pass

    base_model_weights = load_safetensors(models['base']['download_path']) if models['base']['has_safetensors'] else load_pytorch_tensors(models['base']['download_path'])
    finetuned_model_weights = load_safetensors(models['finetuned']['download_path']) if models['finetuned']['has_safetensors'] else load_pytorch_tensors(models['finetuned']['download_path'])
    print("Loaded models")


    loras = {

    }

    # lower rank captures less of the original model, a rank of 32 is probably reasonable for small delta (task specific finetunes and such)
    if not args.lora_rank:
        lora_rank = 64
    else:
        lora_rank = args.lora_rank
    print(f"Decomposing with rank {lora_rank}")
    lora_alpha = args.lora_alpha

    for module in tqdm(linear_module_names):
        target_tensor = finetuned_model_weights[module+".weight"]
        base_tensor = base_model_weights[module+".weight"]
        print(module,target_tensor.shape)
        lora_A, lora_B = decompose_delta_weight(target_tensor, base_tensor, lora_alpha, lora_rank)
        loras[f"base_model.model.{module}.lora_A.weight"] = lora_A.to('cpu')
        loras[f"base_model.model.{module}.lora_B.weight"] = lora_B.to('cpu')

    if not args.output_dir:
        output_dir = os.path.join('./output', f"{args.finetuned_model_name_or_path.split('/')[-1]}_lora_{lora_rank}")
    else:
        output_dir = os.path.join(args.output_dir, ".")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir}")

    lora_config = LoraConfig(
            lora_alpha=lora_alpha, # Setting the alpha to the to decomposition rank value (instead of alpha value used) seems to give better performance. Further testing would be needed to understand what is the optimal alpha value to use
            lora_dropout=0,
            r=lora_rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= list(set([get_module_peft_name(e) for e in linear_module_names])),
    )
    for key in loras.keys():
        loras[key] = loras[key].to('cpu').contiguous()


    model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, state_dict={})

    peft_model = get_peft_model(model, lora_config)

    # Save to disk
    peft_model.save_pretrained(output_dir)
    safetensors.torch.save_file(loras, os.path.join(output_dir, 'adapter_model.safetensors'))
    print(f"LoRA model saved to {args.output_dir}")