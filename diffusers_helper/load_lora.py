from pathlib import Path
from typing import Optional
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers

def load_lora(transformer, lora_path: Path, weight_name: Optional[str] = "pytorch_lora_weights.safetensors", convert_to_diffusers: bool = True):
    """
    Load LoRA weights into the transformer model.
    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path (Path): Path to the LoRA weights file.
        weight_name (Optional[str]): Name of the weight to load.
        convert_to_diffusers (bool): Whether to convert the LoRA weights to diffusers format.
    """

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

    # Convert LoRA weights to diffusers format if requested
    if convert_to_diffusers:
        try:
            print(f"Converting LoRA weights to diffusers format: {weight_name}")
            state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)
            print("LoRA conversion successful")
        except Exception as e:
            print(f"Warning: Could not convert LoRA to diffusers format: {str(e)}")
            print("Proceeding with original format")

    transformer.load_lora_adapter(state_dict, network_alphas=None)
    print("LoRA weights loaded successfully.")
    return transformer