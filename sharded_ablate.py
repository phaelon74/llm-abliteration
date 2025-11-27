import argparse
import gc
import json
import os
import shutil
import threading
import torch
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils import cached_file


def magnitude_sparsify(tensor: torch.Tensor, fraction: float) -> torch.Tensor:
    """Keep only the top fraction of values by magnitude, zero out the rest."""
    if fraction >= 1.0:
        return tensor
    k = int(tensor.numel() * fraction)
    if k == 0:
        return torch.zeros_like(tensor)
    
    flat = tensor.flatten()
    threshold = torch.topk(flat.abs(), k, largest=True, sorted=False)[0].min()
    mask = tensor.abs() >= threshold
    return tensor * mask


"""
A warning regarding PyTorch's convention vs. Safetensors storage:

PyTorch nn.Linear layers store weights as [out_features, in_features] - each row is an output neuron's weights
Safetensors (HuggingFace format) stores them as [in_features, out_features] - transposed!
"""


def modify_tensor_norm_preserved(
    W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0, device_id: int = 0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating refusal direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    
    Args:
        W: Weight tensor to modify
        refusal_dir: Refusal direction vector
        scale_factor: Scaling factor for ablation
        device_id: GPU device ID to use for computation (default: 0)
    """
    original_dtype = W.dtype
    if torch.cuda.is_available() and device_id < torch.cuda.device_count():
        device = f'cuda:{device_id}'
    else:
        device = 'cpu'

    with torch.no_grad():
        # Move tensors for computation on specified device
        # Transpose here to convert from safetensors convention
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure refusal_dir is a 1-dimensional tensor
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        
        # Normalize refusal direction
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)

        # Decompose weight matrix
        # W_gpu is [out_features, in_features]
        W_norm = torch.norm(W_gpu, dim=1, keepdim=True)  # [out_features, 1]
        W_direction = torch.nn.functional.normalize(W_gpu, dim=1)  # normalized per output neuron
    
        # Apply abliteration to the DIRECTIONAL component
        # Compute dot product of each row with refusal direction
        projection = torch.matmul(W_direction, refusal_normalized)  # [in_features]
        
        # Subtract the projection
        W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
    
        # Re-normalize the adjusted direction
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
    
        # Recombine: keep original magnitude, use new direction
        W_modified = W_norm * W_direction_new
        
        # Convert back to original dtype and CPU
        # Transpose here to return safetensors convention
        result = W_modified.T.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup
        del W_gpu, refusal_dir_gpu, refusal_normalized
        del W_direction, W_direction_new, W_norm, projection, W_modified
        
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            # Synchronize and clear cache for the specific device
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    return result.detach().clone()


def process_single_shard(
    shard_file: str,
    shard_path: Path,
    shard_modifications: list,
    measures: dict,
    precision: torch.dtype,
    output_path: str,
    device_id: int,
    file_lock: threading.Lock,
) -> None:
    """
    Process a single shard: load, modify if needed, and save.
    This function is designed to be called in parallel across multiple GPUs.
    
    Args:
        shard_file: Name of the shard file
        shard_path: Full path to the shard file
        shard_modifications: List of (key, layer, measurement, scale, sparsity) tuples
        measures: Dictionary of measurement tensors
        precision: Model precision dtype
        output_path: Output directory path
        device_id: GPU device ID to use
        file_lock: Thread lock for file I/O operations
    """
    if shard_modifications:
        # Shard needs modification
        # Load the entire shard
        state_dict = load_file(str(shard_path))
        
        # Apply all modifications for this shard
        for key, layer, measurement, scale, sparsity in shard_modifications:
            if key in state_dict:
                # Compute refusal direction on-the-fly
                refusal_dir = measures[f'refuse_{measurement}'].float()
                harmless_dir = measures[f'harmless_{layer}'].float()
                
                # Normalize harmless direction
                harmless_normalized = torch.nn.functional.normalize(harmless_dir, dim=0)
                
                # Project and subtract to refine refusal direction
                projection_scalar = refusal_dir @ harmless_normalized
                refined_refusal_dir = refusal_dir - projection_scalar * harmless_normalized
                refusal_dir = refined_refusal_dir.to(precision)
                
                # Apply sparsity
                if sparsity > 0.0:
                    refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)
                
                # Normalize
                refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
                
                # Apply modification using specified GPU
                state_dict[key] = modify_tensor_norm_preserved(
                    state_dict[key],
                    refusal_dir,
                    scale,
                    device_id=device_id,
                ).contiguous()
                
                # Clean up
                del refusal_dir, harmless_dir, harmless_normalized, refined_refusal_dir
                gc.collect()
        
        # Save modified shard with thread-safe file I/O
        output_file_path = os.path.join(output_path, shard_file)
        # Use lock to ensure thread-safe file writing
        with file_lock:
            save_file(state_dict, output_file_path)
        
        # Clean up
        del state_dict
        gc.collect()
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
    else:
        # Just copy unmodified shards (no need to load)
        # Use lock for thread-safe file copying
        output_file_path = os.path.join(output_path, shard_file)
        with file_lock:
            shutil.copy(str(shard_path), output_file_path)


def ablate_by_layers_sharded(
    model_name: str,
    measures: dict,
    marching_orders: list,
    output_path: str,
) -> None:
    """
    Memory-efficient ablation for sharded models with multi-GPU support.
    Handles both local paths and HuggingFace Hub models.
    Processes multiple shards in parallel across available GPUs.
    """
    
    # Detect available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        print(f"Detected {num_gpus} GPU(s) for parallel processing")
    else:
        print("No GPUs detected, using CPU")
    
    # Load config using transformers (handles both local and HF hub)
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    
    # Determine precision
    if hasattr(config, "torch_dtype"):
        precision = config.torch_dtype
    elif hasattr(config, "dtype"):
        precision = config.dtype
    else:
        precision = torch.float32
    
    if isinstance(precision, str):
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        precision = precision_map.get(precision, torch.float32)
    
    print(f"Model precision: {precision}")
    
    # Get the safetensors index file (handles cache)
    index_path = cached_file(model_name, "model.safetensors.index.json")
    model_dir = Path(index_path).parent
    
    print(f"Model directory: {model_dir}")
    
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Find layer prefix
    layer_prefix = None
    for key in weight_map.keys():
        if ".layers." in key and ".self_attn." in key:
            layer_prefix = key.split(".layers.")[0]
            print(f"Detected layer prefix: {layer_prefix}")
            break
    
    if layer_prefix is None:
        raise ValueError("Could not detect layer structure in model weights")
    
    # Build a map of which keys in which shards need modification
    shard_modifications = {}  # shard_file -> [(key, layer, measurement, scale, sparsity)]
    
    for layer, measurement, scale, sparsity in marching_orders:
        # Build the key patterns for this layer
        o_proj_pattern = f"{layer_prefix}.layers.{layer}.self_attn.o_proj.weight"
        down_proj_pattern = f"{layer_prefix}.layers.{layer}.mlp.down_proj.weight"
        
        # Find keys that match
        for key, shard_file in weight_map.items():
            if key == o_proj_pattern or key == down_proj_pattern:
                if shard_file not in shard_modifications:
                    shard_modifications[shard_file] = []
                shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
    
    print(f"\nWill modify {len(shard_modifications)} shards out of {len(set(weight_map.values()))} total")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare shards for processing
    all_shards = sorted(set(weight_map.values()))
    
    # Create thread lock for file I/O operations
    file_lock = threading.Lock()
    
    # Determine number of workers (use number of GPUs, or 1 for CPU)
    max_workers = max(1, num_gpus) if num_gpus > 0 else 1
    
    # Process shards in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for idx, shard_file in enumerate(all_shards):
            shard_path = model_dir / shard_file
            modifications = shard_modifications.get(shard_file, [])
            
            # Assign GPU using round-robin
            device_id = idx % max_workers if num_gpus > 0 else 0
            
            # Submit shard processing task
            future = executor.submit(
                process_single_shard,
                shard_file,
                shard_path,
                modifications,
                measures,
                precision,
                output_path,
                device_id,
                file_lock,
            )
            futures.append((future, shard_file))
        
        # Process with progress bar
        for future, shard_file in tqdm(futures, desc="Processing shards"):
            try:
                future.result()
            except Exception as e:
                print(f"\nError processing {shard_file}: {e}")
                raise
    
    # Copy the index file
    print("\nCopying configuration files...")
    shutil.copy(str(index_path), f"{output_path}/model.safetensors.index.json")
    
    # Copy all config files that exist
    config_files = [
        "config.json", 
        "tokenizer_config.json", 
        "tokenizer.json",
        "special_tokens_map.json", 
        "generation_config.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "preprocessor_config.json",
        "chat_template.json"
    ]
    
    for file in config_files:
        try:
            src_path = cached_file(model_name, file)
            if src_path and os.path.exists(src_path):
                shutil.copy(src_path, f"{output_path}/{file}")
        except Exception:
            # File doesn't exist, skip it
            pass
    
    print(f"\nModified model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient sharded ablation script using YAML configuration."
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to a YAML configuration file',
    )
    
    args = parser.parse_args()
    
    # Load YAML configuration
    with open(args.file_path, 'r') as file:
        ydata = yaml.safe_load(file)
    
    model_name = ydata.get("model")
    measurement_file = ydata.get("measurements")
    output_dir = ydata.get("output")
    ablations = ydata.get("ablate")
    
    print("=" * 60)
    print("SHARDED ABLATION CONFIGURATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Measurements: {measurement_file}")
    print(f"Output directory: {output_dir}")
    print(f"Number of ablations: {len(ablations)}")
    print("=" * 60)
    
    # Load measurements
    print(f"\nLoading measurements from {measurement_file}...")
    measures = torch.load(measurement_file)
    print(f"Loaded {len(measures)} measurements")
    
    # Parse ablation orders
    orders = [
        (
            int(item['layer']),
            int(item['measurement']),
            float(item['scale']),
            float(item['sparsity']),
        )
        for item in ablations
    ]
    
    print("\nAblation orders:")
    for layer, measurement, scale, sparsity in orders:
        print(f"  Layer {layer}: measurement={measurement}, scale={scale}, sparsity={sparsity}")
    
    # Perform sharded ablation
    print("\n" + "=" * 60)
    print("STARTING ABLATION")
    print("=" * 60)
    ablate_by_layers_sharded(
        model_name=model_name,
        measures=measures,
        marching_orders=orders,
        output_path=output_dir,
    )
    
    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
