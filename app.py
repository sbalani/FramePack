from diffusers_helper.hf_login import login

import os
import subprocess
import platform
import random
import time  # Add time module for duration tracking
import shutil
import glob

# Add this line to handle HF download errors
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp, save_bcthw_as_gif, save_bcthw_as_apng, save_bcthw_as_webp, generate_new_timestamp, save_individual_frames
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.load_lora import load_lora
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
args = parser.parse_args()

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

# Change directory paths to use absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
outputs_folder = os.path.join(current_dir, 'outputs')
used_images_folder = os.path.join(outputs_folder, 'used_images')
intermediate_videos_folder = os.path.join(outputs_folder, 'intermediate_videos')
gif_videos_folder = os.path.join(outputs_folder, 'gif_videos')
apng_videos_folder = os.path.join(outputs_folder, 'apng_videos')
webp_videos_folder = os.path.join(outputs_folder, 'webp_videos')
webm_videos_folder = os.path.join(outputs_folder, 'webm_videos')
intermediate_gif_videos_folder = os.path.join(outputs_folder, 'intermediate_gif_videos')
intermediate_apng_videos_folder = os.path.join(outputs_folder, 'intermediate_apng_videos')
intermediate_webp_videos_folder = os.path.join(outputs_folder, 'intermediate_webp_videos')
intermediate_webm_videos_folder = os.path.join(outputs_folder, 'intermediate_webm_videos')
individual_frames_folder = os.path.join(outputs_folder, 'individual_frames')
intermediate_individual_frames_folder = os.path.join(individual_frames_folder, 'intermediate_videos')
last_frames_folder = os.path.join(outputs_folder, 'last_frames')  # Add last_frames folder
intermediate_last_frames_folder = os.path.join(last_frames_folder, 'intermediate_videos')  # Add intermediate last frames folder
loras_folder = os.path.join(current_dir, 'loras')  # Add loras folder

# Ensure all directories exist with proper error handling
for directory in [
    outputs_folder,
    used_images_folder,
    intermediate_videos_folder,
    gif_videos_folder,
    apng_videos_folder,
    webp_videos_folder,
    webm_videos_folder,
    intermediate_gif_videos_folder,
    intermediate_apng_videos_folder,
    intermediate_webp_videos_folder,
    intermediate_webm_videos_folder,
    individual_frames_folder,
    intermediate_individual_frames_folder,
    last_frames_folder,  # Add last_frames folder to the list
    intermediate_last_frames_folder,  # Add intermediate last frames folder to the list
    loras_folder  # Add loras folder to the list
]:
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    except Exception as e:
        print(f"Error creating directory {directory}: {str(e)}")

# Add batch processing output folder
outputs_batch_folder = os.path.join(outputs_folder, 'batch_outputs')
try:
    os.makedirs(outputs_batch_folder, exist_ok=True)
    print(f"Created batch outputs directory: {outputs_batch_folder}")
except Exception as e:
    print(f"Error creating batch outputs directory: {str(e)}")

def open_folder(folder_path):
    """Opens the specified folder in the file explorer/manager in a cross-platform way."""
    try:
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
             return f"Folder does not exist: {folder_path}"
        if platform.system() == "Windows":
            os.startfile(folder_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", folder_path])
        else:  # Linux
            subprocess.run(["xdg-open", folder_path])
        return f"Opened {os.path.basename(folder_path)} folder"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

def print_supported_image_formats():
    """Print information about supported image formats for batch processing."""
    # List of extensions we handle
    extensions = [
        '.png', '.jpg', '.jpeg', '.bmp', '.webp',
        '.tif', '.tiff', '.gif', '.eps', '.ico',
        '.ppm', '.pgm', '.pbm', '.tga', '.exr', '.dib'
    ]

    # Check which formats PIL actually supports in this environment
    supported_formats = []
    unsupported_formats = []

    for ext in extensions:
        format_name = ext[1:].upper()  # Remove dot and convert to uppercase
        if format_name == 'JPG':
            format_name = 'JPEG'  # PIL uses JPEG, not JPG

        try:
            Image.init()
            if format_name in Image.ID or format_name in Image.MIME:
                supported_formats.append(ext)
            else:
                unsupported_formats.append(ext)
        except:
            unsupported_formats.append(ext)

    # print("\nSupported image formats for batch processing:")
    # print(", ".join(supported_formats))

    # if unsupported_formats:
    #     print("\nUnsupported formats in this environment:")
    #     print(", ".join(unsupported_formats))

    return supported_formats

def get_images_from_folder(folder_path):
    """Get all image files from a folder."""
    if not folder_path or not os.path.exists(folder_path):
        return []

    # Get dynamically supported image formats
    image_extensions = print_supported_image_formats()
    images = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in image_extensions:
            images.append(file_path)

    return sorted(images)

def get_prompt_from_txt_file(image_path):
    """Check for a matching txt file with the same name as the image and return its content as prompt."""
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                # Return raw content with newlines preserved, don't strip() the whole content
                content = f.read()
                return content.rstrip()  # Just remove trailing whitespace but keep newlines
        except Exception as e:
            print(f"Error reading prompt file {txt_path}: {str(e)}")
    return None

def format_time_human_readable(seconds):
    """Format time in a human-readable format (e.g., 3 min 11 seconds, 1 hour 12 min 15 seconds)."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} {minutes} min {seconds} seconds"
    elif minutes > 0:
        return f"{minutes} min {seconds} seconds"
    else:
        return f"{seconds} seconds"

def save_processing_metadata(output_path, metadata):
    """Save processing metadata to a text file."""
    metadata_path = os.path.splitext(output_path)[0] + '_metadata.txt'
    try:
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        return True
    except Exception as e:
        print(f"Error saving metadata to {metadata_path}: {str(e)}")
        return False

def move_and_rename_output_file(original_file, target_folder, original_image_filename):
    """Move and rename the output file to match the input image filename."""
    if not original_file or not os.path.exists(original_file):
        return None

    # Get the original extension
    ext = os.path.splitext(original_file)[1]

    # Create the new filename with the same name as the input image
    new_filename = os.path.splitext(original_image_filename)[0] + ext
    new_filepath = os.path.join(target_folder, new_filename)

    try:
        # Ensure target directory exists
        os.makedirs(os.path.dirname(new_filepath), exist_ok=True)

        # Copy instead of move to preserve the original in outputs folder
        import shutil
        shutil.copy2(original_file, new_filepath)
        print(f"Saved output to {new_filepath}")
        return new_filepath
    except Exception as e:
        print(f"Error moving/renaming file to {new_filepath}: {str(e)}")
        return None

# Add function to scan for LoRA files
def scan_lora_files():
    """Scan the loras folder for LoRA files (.safetensors or .pt) and return a list of them."""
    try:
        safetensors_files = glob.glob(os.path.join(loras_folder, "**/*.safetensors"), recursive=True)
        pt_files = glob.glob(os.path.join(loras_folder, "**/*.pt"), recursive=True)
        all_lora_files = safetensors_files + pt_files

        # Format for dropdown: Use basename without extension as display name, full path as value
        lora_options = [("None", "none")]  # Add "None" option
        for lora_file in all_lora_files:
            display_name = os.path.splitext(os.path.basename(lora_file))[0]
            lora_options.append((display_name, lora_file))

        return lora_options
    except Exception as e:
        print(f"Error scanning LoRA files: {str(e)}")
        return [("None", "none")]

def get_lora_path_from_name(display_name):
    """Convert a LoRA display name to its file path."""
    if display_name == "None":
        return "none"

    lora_options = scan_lora_files()
    for name, path in lora_options:
        if name == display_name:
            return path

    # If not found, return none
    print(f"Warning: LoRA '{display_name}' not found in options, using None")
    return "none"

def refresh_loras():
    """Refresh the LoRA dropdown with newly scanned files."""
    lora_options = scan_lora_files()
    return gr.update(choices=[name for name, _ in lora_options], value="None")

def safe_unload_lora(model, device=None):
    """
    Safely unload LoRA weights from the model, handling different model types.

    Args:
        model: The model to unload LoRA weights from
        device: Optional device to move the model to before unloading
    """
    if device is not None:
        model.to(device)

    # Check if this is a DynamicSwap wrapped model
    is_dynamic_swap = 'DynamicSwap' in model.__class__.__name__

    try:
        # First try the standard unload_lora_weights method
        if hasattr(model, "unload_lora_weights"):
            print("Unloading LoRA using unload_lora_weights method")
            model.unload_lora_weights()
            return True
        # Try peft's adapter handling if available
        elif hasattr(model, "peft_config") and model.peft_config:
            if hasattr(model, "disable_adapters"):
                print("Unloading LoRA using disable_adapters method")
                model.disable_adapters()
                return True
            # For PEFT models without disable_adapters method
            elif hasattr(model, "active_adapters") and model.active_adapters:
                print("Clearing active adapters list")
                model.active_adapters = []
                return True
        # Special handling for DynamicSwap models
        elif is_dynamic_swap:
            print("DynamicSwap model detected, attempting to reset internal model state")

            # For DynamicSwap models, try to check if there's an internal model that has LoRA attributes
            if hasattr(model, "model"):
                internal_model = model.model
                if hasattr(internal_model, "unload_lora_weights"):
                    print("Unloading LoRA from internal model")
                    internal_model.unload_lora_weights()
                    return True
                elif hasattr(internal_model, "peft_config") and internal_model.peft_config:
                    if hasattr(internal_model, "disable_adapters"):
                        print("Disabling adapters on internal model")
                        internal_model.disable_adapters()
                        return True

            # If all else fails with DynamicSwap, try to directly remove LoRA modules
            print("Attempting direct LoRA module removal as fallback")
            return force_remove_lora_modules(model)
        else:
            print("No LoRA adapter found to unload")
            return True
    except Exception as e:
        print(f"Error during LoRA unloading: {str(e)}")
        traceback.print_exc()

        # Last resort - try direct module removal
        print("Attempting direct LoRA module removal after error")
        return force_remove_lora_modules(model)

    return False

def force_remove_lora_modules(model):
    """
    Force-remove LoRA modules by directly modifying the model's state.
    This is a last-resort method when normal unloading fails.

    Args:
        model: The model to remove LoRA modules from

    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Look for any LoRA-related modules
        lora_removed = False
        for name, module in list(model.named_modules()):
            # Check for typical PEFT/LoRA module names
            if 'lora' in name.lower():
                print(f"Found LoRA module: {name}")
                lora_removed = True

                # Get parent module and attribute name
                parent_name, _, attr_name = name.rpartition('.')
                if parent_name:
                    try:
                        parent = model.get_submodule(parent_name)
                        if hasattr(parent, attr_name):
                            # Try to restore original module if possible
                            if hasattr(module, 'original_module'):
                                setattr(parent, attr_name, module.original_module)
                                print(f"Restored original module for {name}")
                            # Otherwise just try to reset the module
                            else:
                                print(f"Could not restore original module for {name}")
                    except Exception as e:
                        print(f"Error accessing parent module {parent_name}: {str(e)}")

        # Clear PEFT configuration
        if hasattr(model, "peft_config"):
            model.peft_config = None
            print("Cleared peft_config")
            lora_removed = True

        # Clear LoRA adapter references
        if hasattr(model, "active_adapters"):
            model.active_adapters = []
            print("Cleared active_adapters")
            lora_removed = True

        return lora_removed
    except Exception as e:
        print(f"Error during force LoRA removal: {str(e)}")
        traceback.print_exc()
        return False

print_supported_image_formats()

def save_last_frame(frames, output_dir, filename_base):
    """Save only the last frame from a video frames tensor
    
    Args:
        frames: Tensor of frames in bcthw format
        output_dir: Directory to save the last frame
        filename_base: Base filename to use for the saved frame
    """
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract only the last frame but keep the tensor structure (b,c,t,h,w)
        # where t=1 to be compatible with save_individual_frames function
        last_frame_tensor = frames[:, :, -1:, :, :]
        
        # Use the existing utils function to ensure consistent color processing
        from diffusers_helper.utils import save_individual_frames
        frame_paths = save_individual_frames(last_frame_tensor, output_dir, f"{filename_base}_last", return_frame_paths=True)
        
        if frame_paths and len(frame_paths) > 0:
            print(f"Saved last frame to {frame_paths[0]}")
            return frame_paths[0]
        else:
            print("No frames were saved")
            return None
    except Exception as e:
        print(f"Error saving last frame: {str(e)}")
        traceback.print_exc()
        return None

@torch.no_grad()
def worker(input_image, end_image, prompt, n_prompt, seed, use_random_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality='high', export_gif=False, export_apng=False, export_webp=False, num_generations=1, resolution="640", fps=30, selected_lora="none", lora_scale=1.0, convert_lora=True, save_individual_frames=False, save_intermediate_frames=False, save_individual_frames_frames=False):
    # Declare global variables at the beginning of the function
    global transformer, text_encoder, text_encoder_2, image_encoder, vae

    total_latent_sections = (total_second_length * fps) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # Clean up any previously loaded LoRA at the start of a new worker session
    if hasattr(transformer, "peft_config") and transformer.peft_config:
        print("Cleaning up previous LoRA weights at worker start")
        safe_unload_lora(transformer, gpu)

    current_seed = seed
    all_outputs = {}
    last_used_seed = seed

    # Timing variables
    start_time = time.time()
    generation_times = []

    # Post-processing time estimates (in seconds)
    estimated_vae_time_per_frame = 0.05  # Estimate for VAE decoding per frame
    estimated_save_time = 2.0  # Estimate for saving video to disk

    # If we have past generation data, we can update these estimates
    vae_time_history = []
    save_time_history = []

    for gen_idx in range(num_generations):
        # Track start time for current generation
        gen_start_time = time.time()

        if stream.input_queue.top() == 'end':
            stream.output_queue.push(('end', None))
            print("Worker detected end signal at start of generation")
            return

        # Update seed for this generation
        if use_random_seed:
            current_seed = random.randint(1, 2147483647)
        elif gen_idx > 0:  # increment seed for non-random seeds after first generation
            current_seed += 1

        last_used_seed = current_seed
        stream.output_queue.push(('seed_update', current_seed))

        # Use the new naming scheme for job_id
        job_id = generate_new_timestamp()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Starting generation {gen_idx+1}/{num_generations} with seed {current_seed}...'))))

        try:
            # Clean GPU
            if not high_vram:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )

            # Text encoding
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

            if not high_vram:
                fake_diffusers_current_device(text_encoder, gpu)
                load_model_as_complete(text_encoder_2, target_device=gpu)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # Determine bucket size based on start image
            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=resolution)
            print(f"Found best resolution bucket {width} x {height}")

            # Processing input image (start frame)
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing start frame ...'))))
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            try:
                os.makedirs(used_images_folder, exist_ok=True)
                Image.fromarray(input_image_np).save(os.path.join(used_images_folder, f'{job_id}_start.png'))
                print(f"Saved start image to {os.path.join(used_images_folder, f'{job_id}_start.png')}")
            except Exception as e:
                print(f"Error saving start image: {str(e)}")
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # Processing end image (if provided)
            has_end_image = end_image is not None
            end_image_np = None
            end_image_pt = None
            end_latent = None # Initialize end_latent
            if has_end_image:
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...'))))
                # Use the same bucket size as the start image
                end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
                try:
                    os.makedirs(used_images_folder, exist_ok=True)
                    Image.fromarray(end_image_np).save(os.path.join(used_images_folder, f'{job_id}_end.png'))
                    print(f"Saved end image to {os.path.join(used_images_folder, f'{job_id}_end.png')}")
                except Exception as e:
                    print(f"Error saving end image: {str(e)}")
                end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
                end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)
            start_latent = vae_encode(input_image_pt, vae)
            if has_end_image:
                end_latent = vae_encode(end_image_pt, vae) # Encode end latent

            # CLIP Vision
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            # Encode start image
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # Encode end image and combine if present
            if has_end_image:
                end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
                end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
                # Combine embeddings (simple average)
                image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2.0
                print("Combined start and end frame CLIP vision embeddings.")

            # Dtype conversions (moved after potential combination)
            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)


            # Apply LoRA if selected
            using_lora = False
            previous_lora_loaded = hasattr(transformer, "peft_config") and transformer.peft_config

            if selected_lora != "none":
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Loading LoRA {os.path.basename(selected_lora)}...'))))
                try:
                    lora_path, lora_name = os.path.split(selected_lora)
                    if not lora_path:
                        lora_path = loras_folder

                    if previous_lora_loaded:
                        print("Unloading previously loaded LoRA before loading new one")
                        lora_unload_success = safe_unload_lora(transformer, gpu)

                        if not lora_unload_success and 'DynamicSwap' in transformer.__class__.__name__:
                            print("LoRA unloading failed for DynamicSwap model - need to reload model")
                            if not high_vram:
                                unload_complete_models(
                                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                                )
                                from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
                                transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
                                transformer.high_quality_fp32_output_for_inference = True
                                transformer.requires_grad_(False)
                                if not high_vram:
                                    from diffusers_helper.memory import DynamicSwapInstaller
                                    DynamicSwapInstaller.install_model(transformer, device=gpu)
                                else:
                                    transformer.to(gpu)
                                print("Successfully reloaded transformer model")

                    from diffusers_helper.load_lora import load_lora
                    transformer.to(gpu)
                    
                    # Show conversion status in progress bar
                    if convert_lora:
                        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Converting LoRA {os.path.basename(selected_lora)} to Diffusers format...'))))
                    
                    # Use convert_lora parameter from UI
                    current_transformer = load_lora(transformer, lora_path, lora_name, convert_to_diffusers=convert_lora)

                    print("Verifying all LoRA components are on GPU...")
                    for name, module in transformer.named_modules():
                        if 'lora_' in name.lower():
                            module.to(gpu)

                    for name, param in transformer.named_parameters():
                        if 'lora' in name.lower():
                            if param.device.type != 'cuda':
                                print(f"Force moving LoRA parameter {name} from {param.device} to {gpu}")
                                param.data = param.data.to(gpu)

                    if lora_scale != 1.0:
                        print("LoRA scale not working at the moment")
                        # if hasattr(current_transformer, 'active_adapters') and current_transformer.active_adapters:
                            # for adapter_name in current_transformer.active_adapters:
                                # current_transformer.set_adapter_scale(adapter_name, lora_scale)

                    using_lora = True
                    print(f"Successfully loaded LoRA: {lora_name} with scale: {lora_scale}")
                except Exception as e:
                    print(f"Error loading LoRA {selected_lora}: {str(e)}")
                    traceback.print_exc()

            # Sampling
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Start sampling generation {gen_idx+1}/{num_generations}...'))))

            rnd = torch.Generator("cpu").manual_seed(current_seed)
            num_frames = latent_window_size * 4 - 3

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            # Make latent_paddings a list for indexing
            base_latent_paddings = reversed(range(total_latent_sections))
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            else:
                latent_paddings = list(base_latent_paddings) # Convert iterator to list

            for i, latent_padding in enumerate(latent_paddings):
                # is_last_section determines if we are generating the START of the video (padding 0)
                is_last_section = latent_padding == 0
                # is_first_section determines if we are generating the END of the video (highest padding)
                is_first_section = (i == 0) # Check if it's the first item in the list/iterator

                latent_padding_size = latent_padding * latent_window_size

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    print("Worker detected end signal during latent processing")
                    try:
                        if not high_vram:
                            unload_complete_models(
                                text_encoder, text_encoder_2, image_encoder, vae, transformer
                            )
                    except Exception as cleanup_error:
                        print(f"Error during cleanup: {str(cleanup_error)}")
                    return

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                # Determine clean latents
                clean_latents_pre = start_latent.to(history_latents)
                # Get original post latents from history
                clean_latents_post_orig, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)

                # Use end_latent for the post-latents only during the first section (end frame generation)
                if has_end_image and is_first_section:
                    clean_latents_post = end_latent.to(history_latents)
                    print("Using end_latent for clean_latents_post in the first section.")
                else:
                    clean_latents_post = clean_latents_post_orig

                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)


                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                    if using_lora:
                        print("Ensuring LoRA adapters are on correct device (cuda:0)...")
                        for name, module in transformer.named_modules():
                            if 'lora_' in name.lower():
                                try: module.to(gpu)
                                except Exception as e: print(f"Error moving LoRA module {name}: {str(e)}")

                if teacache_threshold > 0:
                    print(f"Teacache: {teacache_threshold}")
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_threshold)
                else:
                    print("Teacache disabled")
                    transformer.initialize_teacache(enable_teacache=False)

                sampling_start_time = time.time()

                def callback(d):
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)

                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        print("\n" + "="*50)
                        print("USER REQUESTED TO END GENERATION - STOPPING...")
                        print("="*50)
                        raise KeyboardInterrupt('User ends the task.')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)

                    elapsed_time = time.time() - sampling_start_time
                    time_per_step = elapsed_time / current_step if current_step > 0 else 0
                    remaining_steps = steps - current_step
                    eta_seconds = time_per_step * remaining_steps

                    expected_frames = latent_window_size * 4 - 3

                    if current_step == steps:
                        post_processing_eta = expected_frames * estimated_vae_time_per_frame + estimated_save_time
                        eta_seconds = post_processing_eta
                    else:
                        post_processing_eta = expected_frames * estimated_vae_time_per_frame + estimated_save_time
                        eta_seconds += post_processing_eta

                    eta_str = f"{eta_seconds/60:.1f} min" if eta_seconds > 60 else f"{eta_seconds:.1f} sec"
                    total_elapsed = time.time() - start_time
                    elapsed_str = f"{total_elapsed/60:.1f} min" if total_elapsed > 60 else f"{total_elapsed:.1f} sec"

                    hint = f'Sampling {current_step}/{steps} (Gen {gen_idx+1}/{num_generations}, Seed {current_seed})'
                    desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / fps) :.2f} seconds (FPS-{fps}). The video is being extended now ...'
                    time_info = f'Elapsed: {elapsed_str} | ETA: {eta_str}'

                    print(f"\rProgress: {percentage}% | {hint} | {time_info}     ", end="")
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, f"{hint}<br/>{time_info}"))))
                    return

                try:
                    if using_lora:
                        print("Final device check for all tensors before sampling...")
                        devices_found = set()
                        for name, param in transformer.named_parameters():
                            if param.requires_grad or 'lora' in name.lower():
                                devices_found.add(str(param.device))
                                if param.device.type != 'cuda':
                                    print(f"Moving {name} from {param.device} to {gpu}")
                                    param.data = param.data.to(gpu)
                        print(f"Devices found for parameters: {devices_found}")

                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=num_frames,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        num_inference_steps=steps,
                        generator=rnd,
                        prompt_embeds=llama_vec,
                        prompt_embeds_mask=llama_attention_mask,
                        prompt_poolers=clip_l_pooler,
                        negative_prompt_embeds=llama_vec_n,
                        negative_prompt_embeds_mask=llama_attention_mask_n,
                        negative_prompt_poolers=clip_l_pooler_n,
                        device=gpu,
                        dtype=torch.bfloat16,
                        image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents, # Pass the potentially modified clean_latents
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback,
                    )
                except ConnectionResetError as e:
                    print(f"Connection Reset Error caught during sampling: {str(e)}")
                    print("Continuing with the process anyway...")
                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        return
                    empty_shape = (1, 16, latent_window_size, height // 8, width // 8)
                    generated_latents = torch.zeros(empty_shape, dtype=torch.float32).cpu()
                    print("Skipping to next generation due to connection error")
                    break # Skip current generation

                section_time = time.time() - sampling_start_time
                print(f"\nSection completed in {section_time:.2f} seconds")

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                if not high_vram:
                    offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(vae, target_device=gpu)

                vae_start_time = time.time()
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                vae_time = time.time() - vae_start_time
                num_frames_decoded = real_history_latents.shape[2]
                vae_time_per_frame = vae_time / num_frames_decoded if num_frames_decoded > 0 else estimated_vae_time_per_frame
                vae_time_history.append(vae_time_per_frame)
                if len(vae_time_history) > 0:
                    estimated_vae_time_per_frame = sum(vae_time_history) / len(vae_time_history)
                print(f"VAE decoding completed in {vae_time:.2f} seconds ({vae_time_per_frame:.3f} sec/frame)")

                if not high_vram:
                    unload_complete_models()

                is_intermediate = not is_last_section
                if is_intermediate:
                    output_filename = os.path.join(intermediate_videos_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
                    webm_output_filename = os.path.join(intermediate_webm_videos_folder, f'{job_id}_{total_generated_latent_frames}.webm')
                else:
                    output_filename = os.path.join(outputs_folder, f'{job_id}.mp4')  # Use only the timestamp as filename for final output
                    webm_output_filename = os.path.join(webm_videos_folder, f'{job_id}.webm')

                save_start_time = time.time()
                try:
                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    save_bcthw_as_mp4(history_pixels, output_filename, fps=fps, video_quality=video_quality)
                    print(f"Saved MP4 video to {output_filename}")

                    # Save last frame of main video if enabled
                    if save_individual_frames_frames:
                        frames_output_dir = os.path.join(
                            intermediate_last_frames_folder if is_intermediate else last_frames_folder,
                            os.path.splitext(os.path.basename(output_filename))[0]
                        )
                        save_last_frame(history_pixels, frames_output_dir, f"{job_id}_mp4")

                    if video_quality == 'web_compatible':
                        os.makedirs(os.path.dirname(webm_output_filename), exist_ok=True)
                        save_bcthw_as_mp4(history_pixels, webm_output_filename, fps=fps, video_quality=video_quality, format='webm')
                        print(f"Saved WebM video to {webm_output_filename}")
                        
                        # Save last frame of webm video if enabled
                        if save_individual_frames_frames:
                            frames_output_dir = os.path.join(
                                intermediate_last_frames_folder if is_intermediate else last_frames_folder,
                                os.path.splitext(os.path.basename(webm_output_filename))[0]
                            )
                            save_last_frame(history_pixels, frames_output_dir, f"{job_id}_webm")
                        
                    # Save individual frames if enabled
                    if ((is_intermediate and save_intermediate_frames) or 
                        (not is_intermediate and save_individual_frames)):
                        # Create a subfolder with the video filename
                        frames_output_dir = os.path.join(
                            intermediate_individual_frames_folder if is_intermediate else individual_frames_folder,
                            os.path.splitext(os.path.basename(output_filename))[0]
                        )
                        # Use the filename base for individual frames
                        from diffusers_helper.utils import save_individual_frames
                        save_individual_frames(history_pixels, frames_output_dir, job_id)
                        print(f"Saved individual frames to {frames_output_dir}")
                except ConnectionResetError as e:
                    print(f"Connection Reset Error during video saving: {str(e)}")
                    print("Continuing with the process anyway...")
                    output_filename = None
                    webm_output_filename = None
                except Exception as e:
                    print(f"Error saving MP4/WebM video: {str(e)}")
                    output_filename = None
                    webm_output_filename = None

                # Metadata saving logic remains the same (triggered on last section)
                # Check if save_metadata is enabled (passed via process/batch_process)
                # This check needs to be added in the calling functions if not already present
                save_metadata_enabled = True # Assume true for now, should be passed
                if save_metadata_enabled and is_last_section:
                    gen_time = time.time() - gen_start_time
                    generation_time_seconds = int(gen_time)
                    generation_time_formatted = format_time_human_readable(gen_time)

                    metadata = {
                        "Prompt": prompt,
                        "Seed": current_seed,
                        "TeaCache": f"Enabled (Threshold: {teacache_threshold})" if teacache_threshold > 0 else "Disabled",
                        "Video Length (seconds)": total_second_length,
                        "FPS": fps,
                        "Steps": steps,
                        "Distilled CFG Scale": gs,
                        "Resolution": resolution,
                        "Generation Time": generation_time_formatted,
                        "Total Seconds": f"{generation_time_seconds} seconds",
                        "Start Frame Provided": True,
                        "End Frame Provided": has_end_image
                    }

                    if selected_lora != "none":
                        lora_name = os.path.basename(selected_lora)
                        metadata["LoRA"] = lora_name
                        metadata["LoRA Scale"] = lora_scale
                        metadata["LoRA Conversion"] = "Enabled" if convert_lora else "Disabled"

                    if output_filename: save_processing_metadata(output_filename, metadata)
                    if export_gif and os.path.exists(os.path.splitext(output_filename)[0] + '.gif'):
                        save_processing_metadata(os.path.splitext(output_filename)[0] + '.gif', metadata)
                    if export_apng and os.path.exists(os.path.splitext(output_filename)[0] + '.png'):
                        save_processing_metadata(os.path.splitext(output_filename)[0] + '.png', metadata)
                    if export_webp and os.path.exists(os.path.splitext(output_filename)[0] + '.webp'):
                        save_processing_metadata(os.path.splitext(output_filename)[0] + '.webp', metadata)
                    if video_quality == 'web_compatible' and webm_output_filename and os.path.exists(webm_output_filename):
                        save_processing_metadata(webm_output_filename, metadata)


                # Save additional formats
                try:
                    if export_gif:
                        gif_filename = os.path.join(intermediate_gif_videos_folder if is_intermediate else gif_videos_folder, f'{job_id}_{total_generated_latent_frames}.gif' if is_intermediate else f'{job_id}.gif')
                        try:
                            os.makedirs(os.path.dirname(gif_filename), exist_ok=True)
                            save_bcthw_as_gif(history_pixels, gif_filename, fps=fps)
                            print(f"Saved GIF animation to {gif_filename}")
                            
                            # Save last frame from GIF if enabled instead of individual frames
                            if save_individual_frames_frames:
                                frames_output_dir = os.path.join(
                                    intermediate_last_frames_folder if is_intermediate else last_frames_folder,
                                    os.path.splitext(os.path.basename(gif_filename))[0]
                                )
                                save_last_frame(history_pixels, frames_output_dir, f"{job_id}_gif")
                        except Exception as e: print(f"Error saving GIF: {str(e)}")

                    if export_apng:
                        apng_filename = os.path.join(intermediate_apng_videos_folder if is_intermediate else apng_videos_folder, f'{job_id}_{total_generated_latent_frames}.png' if is_intermediate else f'{job_id}.png')
                        try:
                            os.makedirs(os.path.dirname(apng_filename), exist_ok=True)
                            save_bcthw_as_apng(history_pixels, apng_filename, fps=fps)
                            print(f"Saved APNG animation to {apng_filename}")
                            
                            # Save last frame from APNG if enabled instead of individual frames
                            if save_individual_frames_frames:
                                frames_output_dir = os.path.join(
                                    intermediate_last_frames_folder if is_intermediate else last_frames_folder,
                                    os.path.splitext(os.path.basename(apng_filename))[0]
                                )
                                save_last_frame(history_pixels, frames_output_dir, f"{job_id}_apng")
                        except Exception as e: print(f"Error saving APNG: {str(e)}")

                    if export_webp:
                        webp_filename = os.path.join(intermediate_webp_videos_folder if is_intermediate else webp_videos_folder, f'{job_id}_{total_generated_latent_frames}.webp' if is_intermediate else f'{job_id}.webp')
                        try:
                            os.makedirs(os.path.dirname(webp_filename), exist_ok=True)
                            save_bcthw_as_webp(history_pixels, webp_filename, fps=fps)
                            print(f"Saved WebP animation to {webp_filename}")
                            
                            # Save last frame from WebP if enabled instead of individual frames
                            if save_individual_frames_frames:
                                frames_output_dir = os.path.join(
                                    intermediate_last_frames_folder if is_intermediate else last_frames_folder,
                                    os.path.splitext(os.path.basename(webp_filename))[0]
                                )
                                save_last_frame(history_pixels, frames_output_dir, f"{job_id}_webp")
                        except Exception as e: print(f"Error saving WebP: {str(e)}")
                except ConnectionResetError as e:
                    print(f"Connection Reset Error during additional format saving: {str(e)}")
                    print("Continuing with the process anyway...")

                save_time = time.time() - save_start_time
                save_time_history.append(save_time)
                if len(save_time_history) > 0:
                    estimated_save_time = sum(save_time_history) / len(save_time_history)
                print(f"Saving operations completed in {save_time:.2f} seconds")
                print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                stream.output_queue.push(('file', output_filename))

                if is_last_section:
                    break # End of generation loop

            gen_time = time.time() - gen_start_time
            generation_times.append(gen_time)
            avg_gen_time = sum(generation_times) / len(generation_times)
            remaining_gens = num_generations - (gen_idx + 1)
            estimated_remaining_time = avg_gen_time * remaining_gens

            print(f"\nGeneration {gen_idx+1}/{num_generations} completed in {gen_time:.2f} seconds")
            if remaining_gens > 0:
                print(f"Estimated time for remaining generations: {estimated_remaining_time/60:.1f} minutes")
                if using_lora and hasattr(transformer, "peft_config") and transformer.peft_config:
                    print("Cleaning up LoRA weights before next generation")
                    safe_unload_lora(transformer, gpu)

            stream.output_queue.push(('timing', {'gen_time': gen_time, 'avg_time': avg_gen_time, 'remaining_time': estimated_remaining_time}))

        except KeyboardInterrupt as e:
            if str(e) == 'User ends the task.':
                print("\n" + "="*50 + "\nGENERATION ENDED BY USER\n" + "="*50)
                if using_lora and hasattr(transformer, "peft_config") and transformer.peft_config:
                    print("Cleaning up LoRA weights after user interruption")
                    safe_unload_lora(transformer, gpu)
                if not high_vram:
                    print("Unloading models from memory...")
                    unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                stream.output_queue.push(('end', None))
                return
            else: raise
        except ConnectionResetError as e:
            print(f"Connection Reset Error outside main processing loop: {str(e)}")
            print("Trying to continue with next generation...")
            if using_lora and hasattr(transformer, "peft_config") and transformer.peft_config:
                print("Cleaning up LoRA weights after connection error")
                safe_unload_lora(transformer, gpu)
            if not high_vram:
                try: unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                except Exception as cleanup_error: print(f"Error during memory cleanup: {str(cleanup_error)}")
            if gen_idx == num_generations - 1:
                stream.output_queue.push(('end', None))
                return
            continue # Continue to next generation
        except Exception as e:
            print("\n" + "="*50 + f"\nERROR DURING GENERATION: {str(e)}\n" + "="*50)
            traceback.print_exc()
            print("="*50)
            if using_lora and hasattr(transformer, "peft_config") and transformer.peft_config:
                print("Cleaning up LoRA weights after generation error")
                safe_unload_lora(transformer, gpu)
            if not high_vram:
                try: unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                except Exception as cleanup_error: print(f"Error during memory cleanup after exception: {str(cleanup_error)}")
            if gen_idx == num_generations - 1:
                stream.output_queue.push(('end', None))
                return
            continue # Continue to next generation

    total_time = time.time() - start_time
    print(f"\nTotal generation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    if hasattr(transformer, "peft_config") and transformer.peft_config:
        print("Final cleanup of LoRA weights at worker completion")
        safe_unload_lora(transformer, gpu)

    stream.output_queue.push(('final_timing', {'total_time': total_time, 'generation_times': generation_times}))
    stream.output_queue.push(('final_seed', last_used_seed))
    stream.output_queue.push(('end', None))
    return


# Modified process function signature
def process(input_image, end_image, prompt, n_prompt, seed, use_random_seed, num_generations, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality='high', export_gif=False, export_apng=False, export_webp=False, save_metadata=True, resolution="640", fps=30, selected_lora="None", lora_scale=1.0, convert_lora=True, use_multiline_prompts=False, save_individual_frames=False, save_intermediate_frames=False, save_individual_frames_frames=False):
    global stream
    assert input_image is not None, 'No start input image!' # Changed assertion message

    lora_path = get_lora_path_from_name(selected_lora)
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), seed, ''

    # Process multi-line prompts if enabled
    if use_multiline_prompts and prompt.strip():
        # Split prompt by lines, filter empty lines and very short prompts (less than 2 chars)
        prompt_lines = [line.strip() for line in prompt.split('\n')]
        prompt_lines = [line for line in prompt_lines if len(line) >= 2]
        
        if not prompt_lines:
            # If no valid prompts after filtering, use the original prompt
            prompt_lines = [prompt.strip()]
    else:
        # Use the regular prompt as a single line
        prompt_lines = [prompt.strip()]
    
    total_prompts = len(prompt_lines)
    print(f"Processing {total_prompts} prompt(s)")
    
    final_video = None
    
    # Loop through each prompt
    for prompt_idx, current_prompt in enumerate(prompt_lines):
        stream = AsyncStream()
        
        print(f"Processing prompt {prompt_idx+1}/{total_prompts}: {current_prompt}")
        yield None, None, f"Processing prompt {prompt_idx+1}/{total_prompts}", '', gr.update(interactive=False), gr.update(interactive=True), seed, ''
        
        # Pass current prompt to worker
        async_run(worker, input_image, end_image, current_prompt, n_prompt, seed, use_random_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality, export_gif, export_apng, export_webp, num_generations, resolution, fps, lora_path, lora_scale, convert_lora, save_individual_frames, save_intermediate_frames, save_individual_frames_frames)

        output_filename = None
        webm_filename = None
        gif_filename = None
        apng_filename = None
        webp_filename = None
        current_seed = seed
        timing_info = ""
        last_output = None

        while True:
            flag, data = stream.output_queue.next()

            if flag == 'seed_update':
                current_seed = data
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, timing_info

            if flag == 'final_seed':
                current_seed = data

            if flag == 'timing':
                gen_time = data['gen_time']
                avg_time = data['avg_time']
                remaining_time = data['remaining_time']
                eta_str = f"{remaining_time/60:.1f} minutes" if remaining_time > 60 else f"{remaining_time:.1f} seconds"
                timing_info = f"Last generation: {gen_time:.2f}s | Average: {avg_time:.2f}s | ETA: {eta_str}"
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, timing_info

            if flag == 'final_timing':
                total_time = data['total_time']
                timing_info = f"Total generation time: {total_time:.2f}s ({total_time/60:.2f} min)"
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, timing_info

            if flag == 'file':
                output_filename = data
                if output_filename is None:
                    print("Warning: No output file was generated due to an error")
                    yield None, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info
                    continue

                last_output = output_filename
                final_video = output_filename  # Update final video with the latest output
                base_name = os.path.basename(output_filename)
                base_name_no_ext = os.path.splitext(base_name)[0]
                is_intermediate = 'intermediate' in output_filename

                if is_intermediate:
                    webm_filename = os.path.join(intermediate_webm_videos_folder, f"{base_name_no_ext}.webm")
                    gif_filename = os.path.join(intermediate_gif_videos_folder, f"{base_name_no_ext}.gif")
                    apng_filename = os.path.join(intermediate_apng_videos_folder, f"{base_name_no_ext}.png")
                    webp_filename = os.path.join(intermediate_webp_videos_folder, f"{base_name_no_ext}.webp")
                else:
                    webm_filename = os.path.join(webm_videos_folder, f"{base_name_no_ext}.webm")
                    gif_filename = os.path.join(gif_videos_folder, f"{base_name_no_ext}.gif")
                    apng_filename = os.path.join(apng_videos_folder, f"{base_name_no_ext}.png")
                    webp_filename = os.path.join(webp_videos_folder, f"{base_name_no_ext}.webp")

                if not os.path.exists(webm_filename): webm_filename = None
                if not os.path.exists(gif_filename): gif_filename = None
                if not os.path.exists(apng_filename): apng_filename = None
                if not os.path.exists(webp_filename): webp_filename = None

                video_file = output_filename if output_filename is not None else None
                if output_filename is not None and video_quality == 'web_compatible' and webm_filename and os.path.exists(webm_filename):
                    video_file = webm_filename

                prompt_info = f" (Prompt {prompt_idx+1}/{total_prompts})" if use_multiline_prompts else ""
                yield video_file, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info + prompt_info

            if flag == 'progress':
                preview, desc, html = data
                
                # Add prompt info to progress display if using multi-line prompts
                if use_multiline_prompts:
                    if html:
                        # Extract the existing hint from html if possible
                        import re
                        hint_match = re.search(r'>(.*?)<br', html)
                        if hint_match:
                            hint = hint_match.group(1)
                            new_hint = f"{hint} (Prompt {prompt_idx+1}/{total_prompts}: {current_prompt[:30]}{'...' if len(current_prompt) > 30 else ''})"
                            html = html.replace(hint, new_hint)
                    
                    # Add prompt info to the description
                    if desc:
                        desc += f" (Prompt {prompt_idx+1}/{total_prompts})"
                        
                yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info

            if flag == 'end':
                if prompt_idx == len(prompt_lines) - 1:  # If this is the last prompt
                    yield final_video, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed, timing_info
                else:
                    # For intermediate prompts, keep the UI in generation mode
                    yield final_video, gr.update(visible=False), f"Completed prompt {prompt_idx+1}/{total_prompts}", '', gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info
                break

    # Only reach this point if all prompts are processed
    yield final_video, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed, timing_info


# Modified batch_process function signature
def batch_process(input_folder, output_folder, batch_end_frame_folder, prompt, n_prompt, seed, use_random_seed, total_second_length,
                  latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold,
                  video_quality='high', export_gif=False, export_apng=False, export_webp=False,
                  skip_existing=True, save_metadata=True, num_generations=1, resolution="640", fps=30,
                  selected_lora="None", lora_scale=1.0, convert_lora=True, batch_use_multiline_prompts=False, 
                  save_individual_frames=False, save_intermediate_frames=False, save_individual_frames_frames=False):
    global stream

    lora_path = get_lora_path_from_name(selected_lora)

    if not input_folder or not os.path.exists(input_folder):
        return None, f"Input folder does not exist: {input_folder}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""

    if not output_folder:
        output_folder = outputs_batch_folder
    else:
        try: os.makedirs(output_folder, exist_ok=True)
        except Exception as e: return None, f"Error creating output folder: {str(e)}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""

    # Check if end frame folder is provided and exists
    use_end_frames = batch_end_frame_folder and os.path.isdir(batch_end_frame_folder)
    if batch_end_frame_folder and not use_end_frames:
         print(f"Warning: End frame folder provided but not found or not a directory: {batch_end_frame_folder}. Proceeding without end frames.")

    image_files = get_images_from_folder(input_folder)
    if not image_files:
        return None, f"No image files found in {input_folder}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""

    yield None, None, f"Found {len(image_files)} images to process. End frames {'enabled' if use_end_frames else 'disabled'}.", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""

    final_output = None
    current_seed = seed

    for idx, image_path in enumerate(image_files):
        start_image_basename = os.path.basename(image_path)
        output_filename_base = os.path.splitext(start_image_basename)[0]
        
        # Get prompt lines before checking skip_existing
        current_prompt = prompt
        custom_prompt = get_prompt_from_txt_file(image_path)
        if custom_prompt:
            current_prompt = custom_prompt
            print(f"Using custom prompt from txt file for {image_path}: {current_prompt}")

        # Process multi-line prompts if enabled
        if batch_use_multiline_prompts and current_prompt.strip():
            # Split specifically by newline character first, then strip and filter
            potential_lines = current_prompt.split('\n')
            prompt_lines = [line.strip() for line in potential_lines if line.strip()] # Strip and remove empty lines
            prompt_lines = [line for line in prompt_lines if len(line) >= 2] # Filter short lines
            
            if not prompt_lines:
                # If no valid prompts after filtering, use the original prompt as one line
                prompt_lines = [current_prompt.strip()]
        else:
            # Use the regular prompt as a single line
            prompt_lines = [current_prompt.strip()]
        
        total_prompts = len(prompt_lines)
        print(f"DEBUG: Detected prompts: {prompt_lines}")
        print(f"Processing {total_prompts} prompt(s) for image {idx+1}/{len(image_files)}")
        
        # Skip check should look for all expected files when multi-line prompts are enabled
        skip_this_image = False
        if skip_existing:
            if batch_use_multiline_prompts:
                # Check if all generations for all prompt lines exist
                all_expected_files_exist = True
                for p_idx in range(total_prompts):
                    for g_idx in range(1, num_generations + 1):
                        # Use consistent naming with both prompt and generation indices
                        suffix = f"_p{p_idx+1}_g{g_idx}" if num_generations > 1 else f"_p{p_idx+1}"
                        output_check_path = os.path.join(output_folder, f"{output_filename_base}{suffix}.mp4")
                        if not os.path.exists(output_check_path):
                            all_expected_files_exist = False
                            break
                    if not all_expected_files_exist:
                        break
                skip_this_image = all_expected_files_exist
            else:
                # Original check for single prompt
                output_filepath_mp4 = os.path.join(output_folder, f"{output_filename_base}{'_1' if num_generations > 1 else ''}.mp4")
                skip_this_image = os.path.exists(output_filepath_mp4)
        
        if skip_this_image:
            print(f"Skipping {image_path} - output already exists")
            yield None, None, f"Skipping {idx+1}/{len(image_files)}: {start_image_basename} - already processed", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""
            continue

        # Load start image
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB': img = img.convert('RGB')
            input_image = np.array(img)
            if len(input_image.shape) == 2: input_image = np.stack([input_image]*3, axis=2)
            print(f"Loaded start image {image_path} with shape {input_image.shape} and dtype {input_image.dtype}")
        except Exception as e:
            print(f"Error loading start image {image_path}: {str(e)}")
            yield None, None, f"Error processing {idx+1}/{len(image_files)}: {start_image_basename} - {str(e)}", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""
            continue

        # Try to load corresponding end image if folder provided
        current_end_image = None
        end_image_path_str = "None"
        if use_end_frames:
            potential_end_path = os.path.join(batch_end_frame_folder, start_image_basename)
            if os.path.exists(potential_end_path):
                try:
                    end_img = Image.open(potential_end_path)
                    if end_img.mode != 'RGB': end_img = end_img.convert('RGB')
                    current_end_image = np.array(end_img)
                    if len(current_end_image.shape) == 2: current_end_image = np.stack([current_end_image]*3, axis=2)
                    print(f"Loaded matching end image: {potential_end_path}")
                    end_image_path_str = potential_end_path
                except Exception as e:
                    print(f"Error loading end image {potential_end_path}: {str(e)}. Processing without end frame.")
            else:
                 print(f"No matching end frame found for {start_image_basename} in {batch_end_frame_folder}")

        # Loop through each prompt
        for prompt_idx, prompt_line in enumerate(prompt_lines):
            # Reset generation counter for each prompt line
            generation_count_for_image = 0
            # Reset current_seed for each prompt line (important to maintain deterministic behavior)
            if use_random_seed:
                # Get a new random seed for each prompt line
                current_seed = random.randint(1, 2147483647)
                # Update UI with new seed
                yield None, None, f"Using new random seed {current_seed} for prompt {prompt_idx+1}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, ""
            
            prompt_info = ""
            if batch_use_multiline_prompts:
                prompt_info = f" (Prompt {prompt_idx+1}/{total_prompts}: {prompt_line[:30]}{'...' if len(prompt_line) > 30 else ''})"
                
            yield None, None, f"Processing {idx+1}/{len(image_files)}: {start_image_basename} (End: {os.path.basename(end_image_path_str) if current_end_image is not None else 'No'}) with {num_generations} generation(s){prompt_info}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, ""

            gen_start_time_batch = time.time() # Track start time for metadata

            # Reset stream and run worker
            stream = AsyncStream()
            
            # Create individual frames folders for batch output if needed
            batch_individual_frames_folder = None
            batch_intermediate_individual_frames_folder = None
            batch_last_frames_folder = None
            batch_intermediate_last_frames_folder = None
            if batch_save_individual_frames or batch_save_intermediate_frames or batch_save_individual_frames_frames:
                batch_individual_frames_folder = os.path.join(output_folder, 'individual_frames')
                batch_intermediate_individual_frames_folder = os.path.join(batch_individual_frames_folder, 'intermediate_videos')
                batch_last_frames_folder = os.path.join(output_folder, 'last_frames')
                batch_intermediate_last_frames_folder = os.path.join(batch_last_frames_folder, 'intermediate_videos')
                os.makedirs(batch_individual_frames_folder, exist_ok=True)
                os.makedirs(batch_intermediate_individual_frames_folder, exist_ok=True)
                os.makedirs(batch_last_frames_folder, exist_ok=True)
                os.makedirs(batch_intermediate_last_frames_folder, exist_ok=True)
                print(f"Created frames folders for batch output in: {output_folder}")
            
            # Custom function for batch worker that overrides the individual_frames_folder path
            def batch_worker_override(*args, **kwargs):
                # Save original paths
                global individual_frames_folder, intermediate_individual_frames_folder, last_frames_folder, intermediate_last_frames_folder
                
                orig_individual_frames = individual_frames_folder
                orig_intermediate_individual_frames = intermediate_individual_frames_folder
                orig_last_frames = last_frames_folder
                orig_intermediate_last_frames = intermediate_last_frames_folder
                
                # Override with batch paths if they exist
                if batch_individual_frames_folder:
                    individual_frames_folder = batch_individual_frames_folder
                    intermediate_individual_frames_folder = batch_intermediate_individual_frames_folder
                    last_frames_folder = batch_last_frames_folder
                    intermediate_last_frames_folder = batch_intermediate_last_frames_folder
                
                try:
                    # Call original worker
                    result = worker(*args, **kwargs)
                    return result
                finally:
                    # Restore original paths
                    if batch_individual_frames_folder:
                        individual_frames_folder = orig_individual_frames
                        intermediate_individual_frames_folder = orig_intermediate_individual_frames
                        last_frames_folder = orig_last_frames
                        intermediate_last_frames_folder = orig_intermediate_last_frames
            
            async_run(batch_worker_override if (batch_save_individual_frames or batch_save_intermediate_frames or batch_save_individual_frames_frames) else worker, 
                    input_image, current_end_image, prompt_line, n_prompt, current_seed, use_random_seed,
                    total_second_length, latent_window_size, steps, cfg, gs, rs,
                    gpu_memory_preservation, teacache_threshold, video_quality, export_gif,
                    export_apng, export_webp, num_generations=num_generations, resolution=resolution, fps=fps,
                    selected_lora=lora_path, lora_scale=lora_scale, convert_lora=convert_lora,
                    save_individual_frames=batch_save_individual_frames,
                    save_intermediate_frames=batch_save_intermediate_frames,
                    save_individual_frames_frames=batch_save_individual_frames_frames)

            output_filename = None
            last_output = None
            all_outputs = {}

            while True:
                flag, data = stream.output_queue.next()

                if flag == 'seed_update':
                    current_seed = data
                    yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, gr.update()

                if flag == 'file':
                    output_filename = data
                    if output_filename:
                        is_intermediate = 'intermediate' in output_filename
                        if not is_intermediate:
                            generation_count_for_image += 1 # Increment only for final outputs
                            
                            # Add prompt index to the suffix if using multi-line prompts
                            if batch_use_multiline_prompts:
                                # Always use both prompt index and generation index for consistency
                                suffix = f"_p{prompt_idx+1}_g{generation_count_for_image}" if num_generations > 1 else f"_p{prompt_idx+1}"
                            else:
                                suffix = f"_{generation_count_for_image}" if num_generations > 1 else ""
                            
                            # Calculate the base name using the determined suffix AFTER the if/else
                            modified_image_filename_base = f"{output_filename_base}{suffix}"

                            # Move MP4
                            moved_file = move_and_rename_output_file(output_filename, output_folder, f"{modified_image_filename_base}.mp4")
                            if moved_file:
                                output_key = f'mp4_{generation_count_for_image}' if num_generations > 1 else 'mp4'
                                all_outputs[output_key] = moved_file
                                last_output = moved_file
                                final_output = moved_file

                                prompt_status = f" (Prompt {prompt_idx+1}/{total_prompts})" if batch_use_multiline_prompts else ""
                                yield last_output, gr.update(visible=False), f"Processing {idx+1}/{len(image_files)}: {start_image_basename} - Generated video {generation_count_for_image}/{num_generations}{prompt_status}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()

                                # Save metadata
                                if save_metadata:
                                    gen_time = time.time() - gen_start_time_batch
                                    generation_time_seconds = int(gen_time)
                                    generation_time_formatted = format_time_human_readable(gen_time)

                                    metadata = {
                                        "Prompt": prompt_line,
                                        "Seed": current_seed,
                                        "TeaCache": f"Enabled (Threshold: {teacache_threshold})" if teacache_threshold > 0 else "Disabled",
                                        "Video Length (seconds)": total_second_length,
                                        "FPS": fps,
                                        "Steps": steps,
                                        "Distilled CFG Scale": gs,
                                        "Resolution": resolution,
                                        "Generation Time": generation_time_formatted,
                                        "Total Seconds": f"{generation_time_seconds} seconds",
                                        "Start Frame": image_path,
                                        "End Frame": end_image_path_str if current_end_image is not None else "None"
                                    }
                                    if batch_use_multiline_prompts:
                                        metadata["Prompt Number"] = f"{prompt_idx+1}/{total_prompts}"
                                        
                                    if lora_path != "none":
                                        lora_name = os.path.basename(lora_path)
                                        metadata["LoRA"] = lora_name
                                        metadata["LoRA Scale"] = lora_scale
                                        metadata["LoRA Conversion"] = "Enabled" if convert_lora else "Disabled"
                                    save_processing_metadata(moved_file, metadata)

                                # Handle other formats
                                if export_gif:
                                    gif_filename = os.path.splitext(output_filename)[0] + '.gif'
                                    moved_gif = move_and_rename_output_file(gif_filename, output_folder, f"{modified_image_filename_base}.gif")
                                    if moved_gif and save_metadata: save_processing_metadata(moved_gif, metadata)
                                if export_apng:
                                    apng_filename = os.path.splitext(output_filename)[0] + '.png'
                                    moved_apng = move_and_rename_output_file(apng_filename, output_folder, f"{modified_image_filename_base}.png")
                                    if moved_apng and save_metadata: save_processing_metadata(moved_apng, metadata)
                                if export_webp:
                                    webp_filename = os.path.splitext(output_filename)[0] + '.webp'
                                    moved_webp = move_and_rename_output_file(webp_filename, output_folder, f"{modified_image_filename_base}.webp")
                                    if moved_webp and save_metadata: save_processing_metadata(moved_webp, metadata)
                                if video_quality == 'web_compatible':
                                    webm_filename = os.path.splitext(output_filename)[0] + '.webm'
                                    moved_webm = move_and_rename_output_file(webm_filename, output_folder, f"{modified_image_filename_base}.webm")
                                    if moved_webm:
                                        if save_metadata: save_processing_metadata(moved_webm, metadata)
                                        last_output = moved_webm
                                        final_output = moved_webm

                        else: # Intermediate file
                            prompt_status = f" (Prompt {prompt_idx+1}/{total_prompts})" if batch_use_multiline_prompts else ""
                            yield output_filename, gr.update(visible=False), f"Processing {idx+1}/{len(image_files)}: {start_image_basename} - Generating intermediate result{prompt_status}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()


                if flag == 'progress':
                    preview, desc, html = data
                    current_progress = f"Processing {idx+1}/{len(image_files)}: {start_image_basename}"
                    
                    # Add prompt info to the progress text if using multi-line prompts
                    if batch_use_multiline_prompts:
                        prompt_status = f" (Prompt {prompt_idx+1}/{total_prompts})"
                        current_progress += prompt_status
                        
                    if desc: current_progress += f" - {desc}"
                    
                    progress_html = html if html else make_progress_bar_html(0, f"Processing file {idx+1} of {len(image_files)}")
                    
                    # Add prompt info to HTML if using multi-line prompts
                    if batch_use_multiline_prompts and html:
                        import re
                        hint_match = re.search(r'>(.*?)<br', html)
                        if hint_match:
                            hint = hint_match.group(1)
                            new_hint = f"{hint} (Prompt {prompt_idx+1}/{total_prompts})"
                            progress_html = html.replace(hint, new_hint)
                    
                    video_update = last_output if last_output else gr.update()
                    yield video_update, gr.update(visible=True, value=preview), current_progress, progress_html, gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()

                if flag == 'end':
                    video_update = last_output if last_output else gr.update()
                    
                    if prompt_idx == len(prompt_lines) - 1:  # If this is the last prompt for this image
                        yield video_update, gr.update(visible=False), f"Completed {idx+1}/{len(image_files)}: {start_image_basename}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()
                    else:
                        prompt_status = f" (Completed prompt {prompt_idx+1}/{total_prompts}, continuing to next prompt)"
                        yield video_update, gr.update(visible=False), f"Processing {idx+1}/{len(image_files)}: {start_image_basename}{prompt_status}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()
                    break # End loop for this prompt

    yield final_output, gr.update(visible=False), f"Batch processing complete. Processed {len(image_files)} images.", "", gr.update(interactive=True), gr.update(interactive=False), current_seed, ""


def end_process():
    print("\nSending end generation signal...")
    if 'stream' in globals() and stream:
        stream.input_queue.push('end')
        print("End signal sent. Waiting for generation to stop safely...")
    else:
        print("Stream not initialized, cannot send end signal.")
    return gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False) # Update both sets of buttons


quick_prompts = [
    'A character doing some simple body movements.','A talking man.'
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack Improved SECourses App V26 - Start/End Frame - https://www.patreon.com/posts/126855226')
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Single Image"):
                    # Modified Image Input Section
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(sources='upload', type="numpy", label="Start Frame", height=320)
                        with gr.Column():
                            end_image = gr.Image(sources='upload', type="numpy", label="End Frame (Optional)", height=320) # Added end_image

                    prompt = gr.Textbox(label="Prompt", value='', lines=3) # Changed lines to 3 for multi-line
                    use_multiline_prompts = gr.Checkbox(label="Use Multi-line Prompts", value=False, info="Process each line of the prompt as a separate generation")
                    example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                    example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                    with gr.Row():
                        save_metadata = gr.Checkbox(label="Save Processing Metadata", value=True, info="Save processing parameters in a text file alongside each video")
                        save_individual_frames = gr.Checkbox(label="Save Individual Frames", value=False, info="Save each frame of the final video as an individual image")
                        save_intermediate_frames = gr.Checkbox(label="Save Intermediate Frames", value=False, info="Save each frame of intermediate videos as individual images")
                        save_individual_frames_frames = gr.Checkbox(label="Save Last Frame Of Generations", value=False, info="Save only the last frame of each generation to the last_frames folder")

                    with gr.Row():
                        start_button = gr.Button(value="Start Generation", variant='primary')
                        end_button = gr.Button(value="End Generation", interactive=False)

                with gr.Tab("Batch Processing"):
                    batch_input_folder = gr.Textbox(label="Input Folder Path (Start Frames)", info="Folder containing starting images to process")
                    # Added End Frame Folder for Batch
                    batch_end_frame_folder = gr.Textbox(label="End Frame Folder Path (Optional)", info="Folder containing matching end frames (same filename as start frame)")
                    batch_output_folder = gr.Textbox(label="Output Folder Path (optional)", info="Leave empty to use the default batch_outputs folder")
                    batch_prompt = gr.Textbox(label="Default Prompt", value='', lines=3, info="Used if no matching .txt file exists") # Changed lines to 3

                    with gr.Row():
                        batch_skip_existing = gr.Checkbox(label="Skip Existing Files", value=True, info="Skip files that already exist in the output folder")
                        batch_save_metadata = gr.Checkbox(label="Save Processing Metadata", value=True, info="Save processing parameters in a text file alongside each video")
                        batch_use_multiline_prompts = gr.Checkbox(label="Use Multi-line Prompts", value=False, info="Process each line of the prompt as a separate generation")

                    with gr.Row():
                        batch_save_individual_frames = gr.Checkbox(label="Save Individual Frames", value=False, info="Save each frame of the final video as an individual image")
                        batch_save_intermediate_frames = gr.Checkbox(label="Save Intermediate Frames", value=False, info="Save each frame of intermediate videos as individual images")
                        batch_save_individual_frames_frames = gr.Checkbox(label="Save Last Frame Of Generations", value=False, info="Save only the last frame of each generation to the last_frames folder")

                    with gr.Row():
                        batch_start_button = gr.Button(value="Start Batch Processing", variant='primary')
                        batch_end_button = gr.Button(value="End Processing", interactive=False)

                    with gr.Row():
                        open_batch_input_folder = gr.Button(value="Open Start Folder")
                        # Added button for end frame folder
                        open_batch_end_folder = gr.Button(value="Open End Folder")
                        open_batch_output_folder = gr.Button(value="Open Output Folder")


            with gr.Group():
                with gr.Row():
                    num_generations = gr.Slider(label="Number of Generations", minimum=1, maximum=50, value=1, step=1, info="Generate multiple videos in sequence")
                    resolution = gr.Dropdown(label="Resolution", choices=["1440","1320","1200","1080","960","840","720", "640", "480", "320", "240"], value="640", info="Output Resolution (bigger than 640 set more Preserved Memory)")

                with gr.Row():
                    teacache_threshold = gr.Slider(label='TeaCache Threshold', minimum=0.0, maximum=0.5, value=0.15, step=0.01, info='0 = Disabled, 0.15 = Default. Higher values = more caching but potentially less detail.')
                    seed = gr.Number(label="Seed", value=31337, precision=0)
                    use_random_seed = gr.Checkbox(label="Random Seed", value=True, info="Use random seeds instead of incrementing")

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)

                with gr.Row():
                    fps = gr.Slider(label="FPS", minimum=10, maximum=60, value=30, step=1, info="Output Videos FPS - Doesn't impact generation speed")
                    total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)

                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)

                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)

                gr.Markdown("### LoRA Settings")
                with gr.Row():
                    with gr.Column():
                        lora_options = scan_lora_files()
                        selected_lora = gr.Dropdown(label="Select LoRA", choices=[name for name, _ in lora_options], value="None", info="Select a LoRA to apply")
                    with gr.Column():
                        with gr.Row():
                             lora_refresh_btn = gr.Button(value="🔄 Refresh", scale=1)
                             lora_folder_btn = gr.Button(value="📁 Open Folder", scale=1)
                        lora_scale = gr.Slider(label="LoRA Scale", minimum=0.0, maximum=2.0, value=1.0, step=0.01, info="Adjust the strength of the LoRA effect (0-2)")
                
                with gr.Row():
                    convert_lora = gr.Checkbox(label="Convert LoRA to Diffusers Format - Add extra 20 GB GPU Inference Preserved Memory - I am still searching better solution", value=False, info="Enable to convert LoRA weights to the Diffusers format (recommended)")

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=2, maximum=128, value=8, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                def update_memory_for_resolution(res):
                    if res == "1440": return 23
                    if res == "1320": return 21
                    if res == "1200": return 19
                    if res == "1080": return 16
                    elif res == "960": return 14
                    elif res == "840": return 12
                    elif res == "720": return 10
                    elif res == "640": return 8
                    else: return 6
                resolution.change(fn=update_memory_for_resolution, inputs=resolution, outputs=gpu_memory_preservation)

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=True, height=512, loop=True)
            video_info = gr.HTML("<div id='video-info'>Generate a video to see information</div>")
            # Updated Markdown Text
            gr.Markdown('''
            Note on Sampling: Due to inverted sampling, the end part of the video is generated first, and the start part last.
            - **Start Frame Only:** If the start action isn't in the video initially, wait for the full generation.
            - **Start and End Frames:** The model attempts a smooth transition. The end frame's influence appears early in the generation process.
            ''')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            timing_display = gr.Markdown("", label="Time Information", elem_classes='no-generating-animation')

            gr.Markdown("### Folder Options")
            with gr.Row():
                open_outputs_btn = gr.Button(value="Open Generations Folder")
                open_batch_outputs_btn = gr.Button(value="Open Batch Outputs Folder")

            video_quality = gr.Radio(
                label="Video Quality",
                choices=["high", "medium", "low", "web_compatible"],
                value="high",
                info="High: Best quality, Medium: Balanced, Low: Smallest file size, Web Compatible: Best browser compatibility"
            )

            gr.Markdown("### Additional Export Formats")
            gr.Markdown("Select additional formats to export alongside MP4:")
            export_gif = gr.Checkbox(label="Export as GIF", value=False, info="Save animation as GIF (larger file size but widely compatible)")
            export_apng = gr.Checkbox(label="Export as APNG", value=False, info="Save animation as Animated PNG (better quality than GIF but less compatible)")
            export_webp = gr.Checkbox(label="Export as WebP", value=False, info="Save animation as WebP (good balance of quality and file size)")

    lora_refresh_btn.click(fn=refresh_loras, outputs=[selected_lora])
    lora_folder_btn.click(fn=lambda: open_folder(loras_folder), outputs=[gr.Text(visible=False)])

    # Update inputs list for single image processing
    ips = [input_image, end_image, prompt, n_prompt, seed, use_random_seed, num_generations, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality, export_gif, export_apng, export_webp, save_metadata, resolution, fps, selected_lora, lora_scale, convert_lora, use_multiline_prompts, save_individual_frames, save_intermediate_frames, save_individual_frames_frames]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed, timing_display])
    # End button needs to update both sets of start/end buttons
    end_button.click(fn=end_process, outputs=[start_button, end_button, batch_start_button, batch_end_button])

    open_outputs_btn.click(fn=lambda: open_folder(outputs_folder), outputs=gr.Text(visible=False))
    open_batch_outputs_btn.click(fn=lambda: open_folder(outputs_batch_folder), outputs=gr.Text(visible=False))

    # Connect batch folder buttons
    batch_folder_status_text = gr.Text(visible=False) # Shared status text for folder opening
    open_batch_input_folder.click(fn=lambda x: open_folder(x) if x else "No input folder specified", inputs=[batch_input_folder], outputs=[batch_folder_status_text])
    open_batch_end_folder.click(fn=lambda x: open_folder(x) if x else "No end frame folder specified", inputs=[batch_end_frame_folder], outputs=[batch_folder_status_text]) # Added for end frame folder
    open_batch_output_folder.click(fn=lambda x: open_folder(x if x else outputs_batch_folder), inputs=[batch_output_folder], outputs=[batch_folder_status_text])

    # Update inputs list for batch processing
    batch_ips = [batch_input_folder, batch_output_folder, batch_end_frame_folder, batch_prompt, n_prompt, seed, use_random_seed,
                total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
                teacache_threshold, video_quality, export_gif, export_apng, export_webp, batch_skip_existing,
                batch_save_metadata, num_generations, resolution, fps, selected_lora, lora_scale, convert_lora, batch_use_multiline_prompts,
                batch_save_individual_frames, batch_save_intermediate_frames, batch_save_individual_frames_frames]
    batch_start_button.click(fn=batch_process, inputs=batch_ips, outputs=[result_video, preview_image, progress_desc, progress_bar, batch_start_button, batch_end_button, seed, timing_display])
    # End button needs to update both sets of start/end buttons
    batch_end_button.click(fn=end_process, outputs=[start_button, end_button, batch_start_button, batch_end_button])

    video_info_js = """
    function updateVideoInfo() {
        // Select the video element within the specific Gradio output component
        const videoResultDiv = document.querySelector('#result_video'); // Assuming default ID or add one
        if (!videoResultDiv) return;
        const videoElement = videoResultDiv.querySelector('video');

        if (videoElement) {
            const infoDiv = document.getElementById('video-info');
            if (!infoDiv) return;

            // Function to update info, called on loadedmetadata or if already loaded
            const displayInfo = () => {
                if (videoElement.videoWidth && videoElement.videoHeight && videoElement.duration) {
                     const format = videoElement.currentSrc ? videoElement.currentSrc.split('.').pop().toUpperCase() : 'N/A';
                     infoDiv.innerHTML = `<p>Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight} | Duration: ${videoElement.duration.toFixed(2)}s | Format: ${format}</p>`;
                } else {
                     infoDiv.innerHTML = '<p>Loading video info...</p>';
                }
            };

            // Check if metadata is already loaded
            if (videoElement.readyState >= 1) { // HAVE_METADATA or higher
                displayInfo();
            } else {
                // Add event listener if not loaded yet
                videoElement.removeEventListener('loadedmetadata', displayInfo); // Remove previous listeners
                videoElement.addEventListener('loadedmetadata', displayInfo);
            }
        } else {
             const infoDiv = document.getElementById('video-info');
             if (infoDiv) infoDiv.innerHTML = "<div>Generate a video to see information</div>";
        }
    }

    // Use MutationObserver to detect when the video component updates
    const observerCallback = function(mutationsList, observer) {
        for(const mutation of mutationsList) {
            if (mutation.type === 'childList') {
                 // Check if the video element or its container was added/changed
                 const videoResultDiv = document.querySelector('#result_video'); // Re-select if needed
                 if (videoResultDiv && (mutation.target === videoResultDiv || videoResultDiv.contains(mutation.target))) {
                    updateVideoInfo();
                    break; // No need to check other mutations if we found the relevant one
                 }
            }
        }
    };

    // Observe the body for changes, adjust target if needed for performance
    const observer = new MutationObserver(observerCallback);
    observer.observe(document.body, { childList: true, subtree: true });

    // Also run on initial load
    if (document.readyState === 'complete') {
      updateVideoInfo();
    } else {
      document.addEventListener('DOMContentLoaded', updateVideoInfo);
    }
    """
    # Add ID to the result_video component for easier JS selection
    result_video.elem_id = "result_video"
    block.load(None, js=video_info_js)


def get_available_drives():
    """Detect available drives on the system regardless of OS"""
    available_paths = []
    if platform.system() == "Windows":
        import string
        from ctypes import windll
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1: drives.append(f"{letter}:\\") # Use backslash for Windows paths
            bitmask >>= 1
        available_paths = drives
    elif platform.system() == "Darwin": # macOS
         available_paths = ["/", "/Volumes"] # Check root and mounted volumes
    else: # Linux
        available_paths = ["/", "/mnt", "/media"] # Common mount points

    # Filter out paths that don't actually exist
    existing_paths = [p for p in available_paths if os.path.exists(p)]
    print(f"Allowed Gradio paths: {existing_paths}")
    return existing_paths


# Don't add port args when modifying
block.launch(
    share=args.share,
    inbrowser=True,
    allowed_paths=get_available_drives() 
)

print_supported_image_formats()
