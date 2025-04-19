from diffusers_helper.hf_login import login

import os
import subprocess
import platform
import random
import time  # Add time module for duration tracking

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
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp, save_bcthw_as_gif, save_bcthw_as_apng, save_bcthw_as_webp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=7860)
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
    intermediate_webm_videos_folder
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

def open_outputs_folder():
    """Opens the outputs folder in the file explorer/manager in a cross-platform way."""
    try:
        if platform.system() == "Windows":
            os.startfile(outputs_folder)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", outputs_folder])
        else:  # Linux
            subprocess.run(["xdg-open", outputs_folder])
        return "Opened outputs folder"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

def open_folder(folder_path):
    """Opens the specified folder in the file explorer/manager in a cross-platform way."""
    try:
        folder_path = os.path.abspath(folder_path)
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
    
    print("\nSupported image formats for batch processing:")
    print(", ".join(supported_formats))
    
    if unsupported_formats:
        print("\nUnsupported formats in this environment:")
        print(", ".join(unsupported_formats))
    
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
                return f.read().strip()
        except Exception as e:
            print(f"Error reading prompt file {txt_path}: {str(e)}")
    return None

def save_processing_metadata(output_path, metadata):
    """Save processing metadata to a text file."""
    metadata_path = os.path.splitext(output_path)[0] + '_metadata.txt'
    try:
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

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, use_random_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, video_quality='high', export_gif=False, export_apng=False, export_webp=False, num_generations=1):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

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
            return
            
        # Update seed for this generation
        if use_random_seed:
            current_seed = random.randint(1, 2147483647)
        elif gen_idx > 0:  # increment seed for non-random seeds after first generation
            current_seed += 1
            
        last_used_seed = current_seed
        stream.output_queue.push(('seed_update', current_seed))
        
        job_id = generate_timestamp()
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
                fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
                load_model_as_complete(text_encoder_2, target_device=gpu)

            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # Processing input image

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

            try:
                Image.fromarray(input_image_np).save(os.path.join(used_images_folder, f'{job_id}.png'))
                print(f"Saved input image to {os.path.join(used_images_folder, f'{job_id}.png')}")
            except Exception as e:
                print(f"Error saving input image: {str(e)}")

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            start_latent = vae_encode(input_image_pt, vae)

            # CLIP Vision

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # Dtype

            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

            # Sampling

            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Start sampling generation {gen_idx+1}/{num_generations}...'))))

            rnd = torch.Generator("cpu").manual_seed(current_seed)
            num_frames = latent_window_size * 4 - 3

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            latent_paddings = reversed(range(total_latent_sections))

            if total_latent_sections > 4:
                # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
                # items looks better than expanding it when total_latent_sections > 4
                # One can try to remove below trick and just
                # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            for latent_padding in latent_paddings:
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    return

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    transformer.initialize_teacache(enable_teacache=False)

                # Track sampling start time for ETA calculation
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
                    
                    # Calculate ETA for sampling steps
                    elapsed_time = time.time() - sampling_start_time
                    time_per_step = elapsed_time / current_step if current_step > 0 else 0
                    remaining_steps = steps - current_step
                    eta_seconds = time_per_step * remaining_steps
                    
                    # Calculate total frames expected in this video
                    expected_frames = latent_window_size * 4 - 3
                    
                    # Add estimated post-processing time (VAE decoding + saving)
                    if current_step == steps:  # At 100%, show post-processing ETA
                        post_processing_eta = expected_frames * estimated_vae_time_per_frame + estimated_save_time
                        eta_seconds = post_processing_eta
                    else:
                        # Add post-processing time to regular ETA
                        post_processing_eta = expected_frames * estimated_vae_time_per_frame + estimated_save_time
                        eta_seconds += post_processing_eta
                    
                    # Format ETA
                    eta_str = ""
                    if eta_seconds > 60:
                        eta_str = f"{eta_seconds/60:.1f} min"
                    else:
                        eta_str = f"{eta_seconds:.1f} sec"
                    
                    # Total elapsed time for all generations
                    total_elapsed = time.time() - start_time
                    elapsed_str = f"{total_elapsed/60:.1f} min" if total_elapsed > 60 else f"{total_elapsed:.1f} sec"
                    
                    hint = f'Sampling {current_step}/{steps} (Gen {gen_idx+1}/{num_generations}, Seed {current_seed})'
                    desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                    time_info = f'Elapsed: {elapsed_str} | ETA: {eta_str}'
                    
                    # Print to command line
                    print(f"\rProgress: {percentage}% | {hint} | {time_info}     ", end="")
                    
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, f"{hint}<br/>{time_info}"))))
                    return

                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    # shift=3.0,
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
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                # Record the time taken for this section
                section_time = time.time() - sampling_start_time
                print(f"\nSection completed in {section_time:.2f} seconds")

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                if not high_vram:
                    offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(vae, target_device=gpu)

                # Start timing VAE decoding
                vae_start_time = time.time()
                
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                
                # Record VAE decoding time and update estimate
                vae_time = time.time() - vae_start_time
                num_frames_decoded = real_history_latents.shape[2]
                vae_time_per_frame = vae_time / num_frames_decoded if num_frames_decoded > 0 else estimated_vae_time_per_frame
                vae_time_history.append(vae_time_per_frame)
                if len(vae_time_history) > 0:
                    estimated_vae_time_per_frame = sum(vae_time_history) / len(vae_time_history)
                
                print(f"VAE decoding completed in {vae_time:.2f} seconds ({vae_time_per_frame:.3f} sec/frame)")

                if not high_vram:
                    unload_complete_models()

                # Determine if this is an intermediate or final video
                is_intermediate = not is_last_section
                
                # Choose appropriate folder for the MP4 video
                if is_intermediate:
                    output_filename = os.path.join(intermediate_videos_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.mp4')
                    webm_output_filename = os.path.join(intermediate_webm_videos_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.webm')
                else:
                    output_filename = os.path.join(outputs_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.mp4')
                    webm_output_filename = os.path.join(webm_videos_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.webm')

                # Start timing save operations
                save_start_time = time.time()
                
                # Pass video quality to the save function
                try:
                    save_bcthw_as_mp4(history_pixels, output_filename, fps=30, video_quality=video_quality)
                    print(f"Saved MP4 video to {output_filename}")
                    
                    # Also save as WebM if video quality is set to web_compatible
                    if video_quality == 'web_compatible':
                        save_bcthw_as_mp4(history_pixels, webm_output_filename, fps=30, video_quality=video_quality, format='webm')
                        print(f"Saved WebM video to {webm_output_filename}")
                except Exception as e:
                    print(f"Error saving MP4/WebM video: {str(e)}")
                    # Create a default output filename in case of failure
                    output_filename = None
                    webm_output_filename = None

                # Save metadata for final outputs if enabled and this is the last section
                if save_metadata and is_last_section:
                    metadata = {
                        "Prompt": prompt,
                        "Seed": current_seed,
                        "TeaCache": "Enabled" if use_teacache else "Disabled",
                        "Video Length (seconds)": total_second_length,
                        "Steps": steps,
                        "Distilled CFG Scale": gs
                    }
                    save_processing_metadata(output_filename, metadata)
                    
                    # Also save metadata for other formats if they were exported
                    if export_gif and os.path.exists(os.path.splitext(output_filename)[0] + '.gif'):
                        save_processing_metadata(os.path.splitext(output_filename)[0] + '.gif', metadata)
                    
                    if export_apng and os.path.exists(os.path.splitext(output_filename)[0] + '.png'):
                        save_processing_metadata(os.path.splitext(output_filename)[0] + '.png', metadata)
                        
                    if export_webp and os.path.exists(os.path.splitext(output_filename)[0] + '.webp'):
                        save_processing_metadata(os.path.splitext(output_filename)[0] + '.webp', metadata)
                        
                    if video_quality == 'web_compatible' and os.path.exists(webm_output_filename):
                        save_processing_metadata(webm_output_filename, metadata)

                # Save in additional formats if requested
                if export_gif:
                    if is_intermediate:
                        gif_filename = os.path.join(intermediate_gif_videos_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.gif')
                    else:
                        gif_filename = os.path.join(gif_videos_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.gif')
                    try:
                        save_bcthw_as_gif(history_pixels, gif_filename, fps=30)
                        print(f"Saved GIF animation to {gif_filename}")
                    except Exception as e:
                        print(f"Error saving GIF: {str(e)}")

                if export_apng:
                    if is_intermediate:
                        apng_filename = os.path.join(intermediate_apng_videos_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.png')
                    else:
                        apng_filename = os.path.join(apng_videos_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.png')
                    try:
                        save_bcthw_as_apng(history_pixels, apng_filename, fps=30)
                        print(f"Saved APNG animation to {apng_filename}")
                    except Exception as e:
                        print(f"Error saving APNG: {str(e)}")

                if export_webp:
                    if is_intermediate:
                        webp_filename = os.path.join(intermediate_webp_videos_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.webp')
                    else:
                        webp_filename = os.path.join(webp_videos_folder, f'{job_id}_seed{current_seed}_{total_generated_latent_frames}.webp')
                    try:
                        save_bcthw_as_webp(history_pixels, webp_filename, fps=30)
                        print(f"Saved WebP animation to {webp_filename}")
                    except Exception as e:
                        print(f"Error saving WebP: {str(e)}")
                
                # Record save time and update estimate
                save_time = time.time() - save_start_time
                save_time_history.append(save_time)
                if len(save_time_history) > 0:
                    estimated_save_time = sum(save_time_history) / len(save_time_history)
                
                print(f"Saving operations completed in {save_time:.2f} seconds")
                print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                # Push the filename to the output queue for processing
                stream.output_queue.push(('file', output_filename))

                # Check if this is the last section and break the loop if it is
                if is_last_section:
                    break
            
            # Record generation time
            gen_time = time.time() - gen_start_time
            generation_times.append(gen_time)
            avg_gen_time = sum(generation_times) / len(generation_times)
            remaining_gens = num_generations - (gen_idx + 1)
            estimated_remaining_time = avg_gen_time * remaining_gens
            
            print(f"\nGeneration {gen_idx+1}/{num_generations} completed in {gen_time:.2f} seconds")
            if remaining_gens > 0:
                print(f"Estimated time for remaining generations: {estimated_remaining_time/60:.1f} minutes")
            
            # Push timing information to the queue
            stream.output_queue.push(('timing', {'gen_time': gen_time, 'avg_time': avg_gen_time, 'remaining_time': estimated_remaining_time}))
            
        except KeyboardInterrupt as e:
            # Handle the user ending the task gracefully
            if str(e) == 'User ends the task.':
                print("\n" + "="*50)
                print("GENERATION ENDED BY USER")
                print("="*50)
                
                # Make sure we unload models to free memory
                if not high_vram:
                    print("Unloading models from memory...")
                    unload_complete_models(
                        text_encoder, text_encoder_2, image_encoder, vae, transformer
                    )
                
                # Push end message to output queue to ensure UI updates correctly
                stream.output_queue.push(('end', None))
                return
            else:
                # Re-raise if it's not our specific interrupt
                raise
        except:
            traceback.print_exc()

            if not high_vram:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )

    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nTotal generation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Return the last used seed and timing info after all generations are complete
    stream.output_queue.push(('final_timing', {'total_time': total_time, 'generation_times': generation_times}))
    stream.output_queue.push(('final_seed', last_used_seed))
    stream.output_queue.push(('end', None))
    return


def process(input_image, prompt, n_prompt, seed, use_random_seed, num_generations, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, video_quality='high', export_gif=False, export_apng=False, export_webp=False, save_metadata=True):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), seed, ''

    stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, use_random_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, video_quality, export_gif, export_apng, export_webp, num_generations)

    output_filename = None
    webm_filename = None
    gif_filename = None
    apng_filename = None
    webp_filename = None
    current_seed = seed
    timing_info = ""
    last_output = None  # Initialize last_output

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'seed_update':
            current_seed = data
            yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, timing_info

        if flag == 'final_seed':
            current_seed = data
            # Final seed will be returned at the end

        if flag == 'timing':
            gen_time = data['gen_time']
            avg_time = data['avg_time']
            remaining_time = data['remaining_time']
            
            if remaining_time > 60:
                eta_str = f"{remaining_time/60:.1f} minutes"
            else:
                eta_str = f"{remaining_time:.1f} seconds"
                
            timing_info = f"Last generation: {gen_time:.2f}s | Average: {avg_time:.2f}s | ETA: {eta_str}"
            yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, timing_info
        
        if flag == 'final_timing':
            total_time = data['total_time']
            timing_info = f"Total generation time: {total_time:.2f}s ({total_time/60:.2f} min)"
            yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, timing_info

        if flag == 'file':
            output_filename = data
            
            # Check if output_filename is None (indicating a save error)
            if output_filename is None:
                print("Warning: No output file was generated due to an error")
                yield None, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info
                continue
                
            last_output = output_filename  # Update last_output
            base_name = os.path.basename(output_filename)
            base_name_no_ext = os.path.splitext(base_name)[0]
            
            # Get the correct paths for all possible output files
            is_intermediate = 'intermediate' in output_filename
            
            # Check for WebM format (in dedicated folder)
            if is_intermediate:
                webm_filename = os.path.join(intermediate_webm_videos_folder, f"{base_name_no_ext}.webm")
            else:
                webm_filename = os.path.join(webm_videos_folder, f"{base_name_no_ext}.webm")
            
            # Determine paths for other formats based on whether this is intermediate or final
            if is_intermediate:
                gif_filename = os.path.join(intermediate_gif_videos_folder, f"{base_name_no_ext}.gif")
                apng_filename = os.path.join(intermediate_apng_videos_folder, f"{base_name_no_ext}.png")
                webp_filename = os.path.join(intermediate_webp_videos_folder, f"{base_name_no_ext}.webp")
            else:
                gif_filename = os.path.join(gif_videos_folder, f"{base_name_no_ext}.gif")
                apng_filename = os.path.join(apng_videos_folder, f"{base_name_no_ext}.png")
                webp_filename = os.path.join(webp_videos_folder, f"{base_name_no_ext}.webp")
            
            # Check if these files were created and exist
            if not os.path.exists(webm_filename):
                webm_filename = None
            if not os.path.exists(gif_filename):
                gif_filename = None
            if not os.path.exists(apng_filename):
                apng_filename = None
            if not os.path.exists(webp_filename):
                webp_filename = None
                
            # Select the appropriate video file based on quality setting
            video_file = output_filename if output_filename is not None else None
            if output_filename is not None and video_quality == 'web_compatible' and webm_filename and os.path.exists(webm_filename):
                video_file = webm_filename
            
            # Metadata is now saved in the worker function
            # This removes duplicate code and ensures metadata is only saved for final outputs
                
            if video_file:
                last_output = video_file  # Ensure last_output is updated
            
            # Use video_file directly to avoid referencing last_output if it's not defined
            yield video_file, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed, timing_info
            break

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info

        if flag == 'end':
            # Select the appropriate video file based on quality setting
            video_file = output_filename if output_filename is not None else None
            if output_filename is not None and video_quality == 'web_compatible' and webm_filename and os.path.exists(webm_filename):
                video_file = webm_filename
            
            # Metadata is now saved in the worker function
            # This removes duplicate code and ensures metadata is only saved for final outputs
                
            if video_file:
                last_output = video_file  # Ensure last_output is updated
            
            # Use video_file directly to avoid referencing last_output if it's not defined
            yield video_file, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed, timing_info
            break


def batch_process(input_folder, output_folder, prompt, n_prompt, seed, use_random_seed, total_second_length, 
                  latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, 
                  video_quality='high', export_gif=False, export_apng=False, export_webp=False, 
                  skip_existing=True, save_metadata=True, num_generations=1):
    global stream
    
    # Check input folder
    if not input_folder or not os.path.exists(input_folder):
        return None, f"Input folder does not exist: {input_folder}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""
    
    # Set default output folder if not provided
    if not output_folder:
        output_folder = outputs_batch_folder
    else:
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            return None, f"Error creating output folder: {str(e)}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""
    
    # Get all images from the input folder
    image_files = get_images_from_folder(input_folder)
    
    if not image_files:
        return None, f"No image files found in {input_folder}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""
    
    yield None, None, f"Found {len(image_files)} images to process", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""
    
    # Process each image
    for idx, image_path in enumerate(image_files):
        # Check if we should skip this file
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + ".mp4"
        output_filepath = os.path.join(output_folder, output_filename)
        
        if skip_existing and os.path.exists(output_filepath):
            print(f"Skipping {image_path} - output already exists: {output_filepath}")
            yield None, None, f"Skipping {idx+1}/{len(image_files)}: {os.path.basename(image_path)} - already processed", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""
            continue
        
        # Check for a custom prompt in a txt file
        current_prompt = prompt
        custom_prompt = get_prompt_from_txt_file(image_path)
        if custom_prompt:
            current_prompt = custom_prompt
            print(f"Using custom prompt from txt file for {image_path}: {current_prompt}")
        
        # Reset the stream for this image
        stream = AsyncStream()
        
        # Load the image
        try:
            # Open image and convert to RGB mode to ensure consistency 
            # (some formats like RGBA PNG, CMYK TIFF, etc. need conversion)
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            input_image = np.array(img)
            
            # Handle grayscale images that might be read as 2D arrays
            if len(input_image.shape) == 2:
                input_image = np.stack([input_image, input_image, input_image], axis=2)
                
            print(f"Loaded image {image_path} with shape {input_image.shape} and dtype {input_image.dtype}")
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            yield None, None, f"Error processing {idx+1}/{len(image_files)}: {os.path.basename(image_path)} - {str(e)}", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""
            continue
        
        # Generate video(s) for this image using the specified number of generations
        yield None, None, f"Processing {idx+1}/{len(image_files)}: {os.path.basename(image_path)} with {num_generations} generation(s)", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""
        
        # Start the worker
        async_run(worker, input_image, current_prompt, n_prompt, seed, use_random_seed, 
                 total_second_length, latent_window_size, steps, cfg, gs, rs, 
                 gpu_memory_preservation, use_teacache, video_quality, export_gif, 
                 export_apng, export_webp, num_generations=num_generations)
        
        # Get the results
        output_filename = None
        current_seed = seed
        last_output = None
        all_outputs = {}  # To store all output files (mp4, gif, etc.)
        
        while True:
            flag, data = stream.output_queue.next()
            
            if flag == 'seed_update':
                current_seed = data
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, gr.update()
            
            if flag == 'file':
                output_filename = data
                
                if output_filename:
                    # Check if this is an intermediate generation (not the final output)
                    is_intermediate = 'intermediate' in output_filename
                    
                    # Only process final outputs for batch folder
                    if not is_intermediate:
                        # Rename and move the file to the batch output folder
                        image_filename = os.path.basename(image_path)
                        
                        # Track which generation number we're on by checking how many outputs we've already collected
                        generation_count = len([k for k in all_outputs.keys() if k.startswith('mp4')])
                        
                        # Add suffix for generation count if generating multiple videos per image
                        if num_generations > 1:
                            # For the base name, add a suffix like _1, _2, etc.
                            base_name = os.path.splitext(image_filename)[0]
                            ext = os.path.splitext(image_filename)[1]
                            # Generation count is 0-based in our tracking, but display as 1-based
                            suffix = f"_{generation_count + 1}" 
                            modified_image_filename = f"{base_name}{suffix}{ext}"
                        else:
                            modified_image_filename = image_filename
                        
                        moved_file = move_and_rename_output_file(output_filename, output_folder, modified_image_filename)
                        
                        if moved_file:
                            output_key = f'mp4_{generation_count}' if num_generations > 1 else 'mp4'
                            all_outputs[output_key] = moved_file
                            last_output = moved_file
                            
                            # Save metadata for this generation right after moving the file
                            if save_metadata:
                                metadata = {
                                    "Prompt": current_prompt,
                                    "Seed": current_seed,
                                    "TeaCache": "Enabled" if use_teacache else "Disabled",
                                    "Video Length (seconds)": total_second_length,
                                    "Steps": steps,
                                    "Distilled CFG Scale": gs
                                }
                                save_processing_metadata(moved_file, metadata)
                        
                        # Also handle other formats if enabled
                        if export_gif:
                            gif_filename = os.path.splitext(output_filename)[0] + '.gif'
                            if os.path.exists(gif_filename):
                                moved_gif = move_and_rename_output_file(gif_filename, output_folder, modified_image_filename)
                                if moved_gif and save_metadata:
                                    gif_key = f'gif_{generation_count}' if num_generations > 1 else 'gif'
                                    all_outputs[gif_key] = moved_gif
                                    # Save metadata for the GIF file
                                    save_processing_metadata(moved_gif, metadata)
                        
                        if export_apng:
                            apng_filename = os.path.splitext(output_filename)[0] + '.png'
                            if os.path.exists(apng_filename):
                                moved_apng = move_and_rename_output_file(apng_filename, output_folder, modified_image_filename)
                                if moved_apng and save_metadata:
                                    apng_key = f'apng_{generation_count}' if num_generations > 1 else 'apng'
                                    all_outputs[apng_key] = moved_apng
                                    # Save metadata for the APNG file
                                    save_processing_metadata(moved_apng, metadata)
                        
                        if export_webp:
                            webp_filename = os.path.splitext(output_filename)[0] + '.webp'
                            if os.path.exists(webp_filename):
                                moved_webp = move_and_rename_output_file(webp_filename, output_folder, modified_image_filename)
                                if moved_webp and save_metadata:
                                    webp_key = f'webp_{generation_count}' if num_generations > 1 else 'webp'
                                    all_outputs[webp_key] = moved_webp
                                    # Save metadata for the WebP file
                                    save_processing_metadata(moved_webp, metadata)
                        
                        # Also handle WebM format
                        if video_quality == 'web_compatible':
                            webm_filename = os.path.splitext(output_filename)[0] + '.webm'
                            if os.path.exists(webm_filename):
                                moved_webm = move_and_rename_output_file(webm_filename, output_folder, modified_image_filename)
                                if moved_webm and save_metadata:
                                    webm_key = f'webm_{generation_count}' if num_generations > 1 else 'webm'
                                    all_outputs[webm_key] = moved_webm
                                    # Save metadata for the WebM file
                                    save_processing_metadata(moved_webm, metadata)
                                    last_output = moved_webm
            
            if flag == 'progress':
                preview, desc, html = data
                current_progress = f"Processing {idx+1}/{len(image_files)}: {os.path.basename(image_path)}"
                if desc:
                    current_progress += f" - {desc}"
                progress_html = html
                if not progress_html:
                    progress_html = make_progress_bar_html(0, f"Processing file {idx+1} of {len(image_files)}")
                yield gr.update(), gr.update(visible=True, value=preview), current_progress, progress_html, gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()
            
            if flag == 'end':
                # All processing for this image is complete
                yield last_output, gr.update(visible=False), f"Completed {idx+1}/{len(image_files)}: {os.path.basename(image_path)}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()
                break
    
    # All images processed
    yield last_output, gr.update(visible=False), f"Batch processing complete. Processed {len(image_files)} images.", "", gr.update(interactive=True), gr.update(interactive=False), current_seed, ""


def end_process():
    print("\nSending end generation signal...")
    stream.input_queue.push('end')
    print("End signal sent. Waiting for generation to stop safely...")
    return gr.update(interactive=True), gr.update(interactive=False)


quick_prompts = [
    'A character doing some simple body movements.','An epic fantasy animation capturing a dramatic confrontation in a swirling desert sandstorm. A heavily armored warrior, helmeted and with a flowing cape, charges or braces himself against a colossal, terrifying monster emerging from the sand. The monster, resembling a golem made of rock and sand, has a skull-like face with a gaping maw filled with sharp teeth. Sand and dust blow violently around both figures, creating a chaotic and intense atmosphere. The warriors cape billows wildly. The monster looms large, possibly roaring or shifting slightly as sand particles constantly stream off its form. Focus on dynamic movement, swirling sand effects, cinematic lighting, and hyperrealistic textures. High detail, fantasy art style.'
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack Improved SECourses App V12 - https://www.patreon.com/posts/126855226')
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Single Image"):
                    input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                    prompt = gr.Textbox(label="Prompt", value='')
                    example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                    example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                    with gr.Row():
                        save_metadata = gr.Checkbox(label="Save Processing Metadata", value=True, info="Save processing parameters in a text file alongside each video")

                    with gr.Row():
                        start_button = gr.Button(value="Start Generation", variant='primary')
                        end_button = gr.Button(value="End Generation", interactive=False)

                with gr.Tab("Batch Processing"):
                    batch_input_folder = gr.Textbox(label="Input Folder Path", info="Folder containing images to process")
                    batch_output_folder = gr.Textbox(label="Output Folder Path (optional)", info="Leave empty to use the default batch_outputs folder")
                    batch_prompt = gr.Textbox(label="Default Prompt", value='', info="Used if no matching .txt file exists")
                    
                    with gr.Row():
                        batch_skip_existing = gr.Checkbox(label="Skip Existing Files", value=True, info="Skip files that already exist in the output folder")
                        batch_save_metadata = gr.Checkbox(label="Save Processing Metadata", value=True, info="Save processing parameters in a text file alongside each video")
                    
                    with gr.Row():
                        batch_start_button = gr.Button(value="Start Batch Processing", variant='primary')
                        batch_end_button = gr.Button(value="End Processing", interactive=False)
                    
                    with gr.Row():
                        open_batch_input_folder = gr.Button(value="Open Input Folder")
                        open_batch_output_folder = gr.Button(value="Open Output Folder")

            with gr.Group():
                # Add number of generations slider
                num_generations = gr.Slider(label="Number of Generations", minimum=1, maximum=50, value=1, step=1, info="Generate multiple videos in sequence")
                
                # Group seed controls in one row
                with gr.Row():
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                    seed = gr.Number(label="Seed", value=31337, precision=0)
                    use_random_seed = gr.Checkbox(label="Random Seed", value=True, info="Use random seeds instead of incrementing")

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                
                # Put Steps and Distilled CFG Scale in the same row
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                


        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=True, height=512, loop=True)
            video_info = gr.HTML("<div id='video-info'>Generate a video to see information</div>")
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            timing_display = gr.Markdown("", label="Time Information", elem_classes='no-generating-animation')
            
            # Add folder navigation buttons
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
    
    # Add JavaScript to show video information when loaded
    video_info_js = """
    function updateVideoInfo() {
        const videoElement = document.querySelector('video');
        if (videoElement) {
            const info = document.getElementById('video-info');
            videoElement.addEventListener('loadedmetadata', function() {
                info.innerHTML = `<p>Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight} | 
                                   Duration: ${videoElement.duration.toFixed(2)}s | 
                                   Format: ${videoElement.currentSrc.split('.').pop()}</p>`;
            });
        }
    }
    
    // Run when page loads and after video updates
    document.addEventListener('DOMContentLoaded', updateVideoInfo);
    const observer = new MutationObserver(function(mutations) {
        updateVideoInfo();
    });
    
    observer.observe(document.documentElement, {
        childList: true,
        subtree: true
    });
    """
    
    block.load(None, js=video_info_js)
    
    # Update inputs list to include new parameters
    ips = [input_image, prompt, n_prompt, seed, use_random_seed, num_generations, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, video_quality, export_gif, export_apng, export_webp, save_metadata]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed, timing_display])
    end_button.click(fn=end_process, outputs=[start_button, end_button])
    
    # Connect folder buttons
    open_outputs_btn.click(fn=lambda: open_folder(outputs_folder), outputs=gr.Text(visible=False))
    open_batch_outputs_btn.click(fn=lambda: open_folder(outputs_batch_folder), outputs=gr.Text(visible=False))
    
    # Connect batch processing buttons
    batch_input_folder_text = gr.Text(visible=False)
    batch_output_folder_text = gr.Text(visible=False)
    
    open_batch_input_folder.click(fn=lambda x: open_folder(x) if x else "No input folder specified", 
                                 inputs=[batch_input_folder], 
                                 outputs=[batch_input_folder_text])
    
    open_batch_output_folder.click(fn=lambda x: open_folder(x if x else outputs_batch_folder), 
                                   inputs=[batch_output_folder], 
                                   outputs=[batch_output_folder_text])
    
    # Connect batch processing start and end buttons
    batch_ips = [batch_input_folder, batch_output_folder, batch_prompt, n_prompt, seed, use_random_seed, 
                total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, 
                use_teacache, video_quality, export_gif, export_apng, export_webp, batch_skip_existing, 
                batch_save_metadata, num_generations]
    
    batch_start_button.click(fn=batch_process, inputs=batch_ips, 
                            outputs=[result_video, preview_image, progress_desc, progress_bar, 
                                     batch_start_button, batch_end_button, seed, timing_display])
    
    batch_end_button.click(fn=end_process, outputs=[batch_start_button, batch_end_button])


def get_available_drives():
    """Detect available drives on the system regardless of OS"""
    available_paths = []
    
    if platform.system() == "Windows":
        import string
        from ctypes import windll
        
        # Check each drive letter
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(f"{letter}:/")
            bitmask >>= 1
            
        available_paths = drives
    else:
        # For Linux/Mac, just use root
        available_paths = ["/"]
        
    print(f"Available drives detected: {available_paths}")
    return available_paths

# Launch with dynamically detected drives
block.launch(
    share=args.share,
    inbrowser=True,
    allowed_paths=get_available_drives()
)

# Print supported image formats at startup for user reference
print("\n=== BATCH PROCESSING INFORMATION ===")
print_supported_image_formats()
print("Place text files with the same name as image files to use as custom prompts.")
print("For example: image1.png and image1.txt for a custom prompt for image1.png")
print("===================================\n")
