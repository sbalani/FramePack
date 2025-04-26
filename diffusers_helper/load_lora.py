from pathlib import Path
from typing import Dict, List, Optional, Union
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
# --- VRAM Debug Import START ---
import torch
from diffusers_helper.memory import get_cuda_free_memory_gb, gpu
# --- VRAM Debug Import END ---

def load_lora(transformer, lora_path: Path, weight_name: Optional[str] = "pytorch_lora_weights.safetensors"):
    """
    Load LoRA weights into the transformer model as a separate adapter.

    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path (Path): Path to the LoRA weights file.
        weight_name (Optional[str]): Name of the weight to load.

    """
    # --- VRAM Debug START ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear cache for more accurate reading
        free_mem_before = get_cuda_free_memory_gb(gpu)
        print(f"[LoRA Load Debug] VRAM Free Before Load: {free_mem_before:.3f} GB")
    # --- VRAM Debug END ---

    state_dict = _fetch_state_dict(
    lora_path,
    weight_name,
    True,
    True,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None)


    state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
    
    adapter_name = weight_name.split(".")[0]
    transformer.load_lora_adapter(state_dict, network_alphas=None, adapter_name=adapter_name)
    print(f"LoRA adapter '{adapter_name}' loaded successfully.")

    # --- VRAM Debug START ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Clear cache again
        free_mem_after = get_cuda_free_memory_gb(gpu)
        vram_used = free_mem_before - free_mem_after
        print(f"[LoRA Load Debug] VRAM Free After Load:  {free_mem_after:.3f} GB")
        print(f"[LoRA Load Debug] VRAM Used by LoRA Load: {vram_used:.3f} GB")
    # --- VRAM Debug END ---

    return transformer
    
# TODO(neph1): remove when HunyuanVideoTransformer3DModelPacked is in _SET_ADAPTER_SCALE_FN_MAPPING
def set_adapters(
        transformer,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):

    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

    # Expand weights into a list, one entry per adapter
    # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)

    if len(adapter_names) != len(weights):
        raise ValueError(
            f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
        )

    # Set None values to default of 1.0
    # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
    weights = [w if w is not None else 1.0 for w in weights]

    # e.g. [{...}, 7] -> [{expanded dict...}, 7]
    scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING["HunyuanVideoTransformer3DModel"]
    weights = scale_expansion_fn(transformer, weights)

    set_weights_and_activate_adapters(transformer, adapter_names, weights)