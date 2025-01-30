import os
import torch
import json

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from .stabledelight.pipeline_yoso_delight import YosoDelightPipeline
from .stabledelight.controlnetvae import ControlNetVAEModel

from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
)

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))
import logging
log = logging.getLogger(__name__)
    
#region Model loading
        
class DownloadAndLoadStableXModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["yoso-delight-v0-4-base",
                           "yoso-normal-v1-8-1"],),
            },
        }

    RETURN_TYPES = ("YOSOPIPE",)
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "loadmodel"
    CATEGORY = "StableXWrapper"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        download_path = os.path.join(folder_paths.models_dir,"diffusers")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=f"Stable-X/{model}",
                #allow_patterns=[f"*{model}*"],
                ignore_patterns=["*text_encoder*", "tokenizer*", "*scheduler*"],
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        torch_dtype = torch.float16
        config_path = os.path.join(model_path, 'unet', 'config.json')
        unet_ckpt_path_safetensors = os.path.join(model_path, 'unet','diffusion_pytorch_model.fp16.safetensors')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)

        with init_empty_weights():
            unet = UNet2DConditionModel(**config)

        if os.path.exists(unet_ckpt_path_safetensors):
            import safetensors.torch
            unet_sd = safetensors.torch.load_file(unet_ckpt_path_safetensors)
        else:
            raise FileNotFoundError(f"No checkpoint found at {unet_ckpt_path_safetensors}")

        for name, param in unet.named_parameters():
            set_module_tensor_to_device(unet, name, device=offload_device, dtype=torch_dtype, value=unet_sd[name])

        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", variant="fp16", device=device, torch_dtype=torch_dtype)
        controlnet = ControlNetVAEModel.from_pretrained(model_path, subfolder="controlnet", variant="fp16", device=device, torch_dtype=torch_dtype)

        empty_text_embedding=torch.load(os.path.join(script_directory, 'empty_text_embed_delight.pt'), map_location=device).to(torch_dtype)

        pipeline = YosoDelightPipeline(
            unet=unet,
            vae = vae,
            controlnet = controlnet,
            empty_text_embedding=empty_text_embedding,
            device=device,
            dtype=torch_dtype,
            pred_type="normal" if "normal" in model else "delight",
            )

        #pipeline.enable_model_cpu_offload()
        return (pipeline,)

class StableXProcessImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("YOSOPIPE",),
                "image": ("IMAGE", ),
                "processing_resolution": ("INT", {"default": 2048, "min": 64, "max": 4096, "step": 16}),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01, "tooltip": "controlnet condition scale"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed only affects normal prediction mode"}),
        },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "StableXWrapper"

    def process(self, pipeline, image, processing_resolution,controlnet_strength, seed):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        image = image.permute(0, 3, 1, 2).to(device).to(torch.float16)

        pipeline.unet.to(device)
        pipeline.vae.to(device)
        pipeline.controlnet.to(device)

        pipe_out = pipeline(
        image,
        controlnet_conditioning_scale=controlnet_strength,
        processing_resolution=processing_resolution,
        generator = torch.Generator(device=device).manual_seed(seed),
        output_type="pt",
        )
        pipeline.unet.to(offload_device)
        pipeline.vae.to(offload_device)
        pipeline.controlnet.to(offload_device)
        pipe_out = (pipe_out.prediction.clip(-1, 1) + 1) / 2

        out_tensor = pipe_out.permute(0, 2, 3, 1).cpu().float()
        
        return (out_tensor, )
    
class DifferenceExtractorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "amplification": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_luminosity_difference"
    CATEGORY = "iStableXWrapper"

    def extract_luminosity_difference(self, original_image, processed_image, amplification=1.0):
        import torch

        # RGB to luminosity conversion weights
        rgb_weights = torch.tensor([0.2126, 0.7152, 0.0722]).to(original_image.device)
        
        # Convert images to luminosity (shape: B,H,W)
        original_lum = torch.sum(original_image * rgb_weights[None, None, None, :], dim=3)
        processed_lum = torch.sum(processed_image * rgb_weights[None, None, None, :], dim=3)
        
        # Calculate luminosity difference
        difference = (original_lum - processed_lum) * amplification
        
        # Normalize and clamp
        difference = torch.clamp(difference, 0, 1)
        
        # Convert back to RGB format (all channels identical)
        difference = difference.unsqueeze(3).repeat(1, 1, 1, 3)
        
        return (difference,)
    
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadStableXModel": DownloadAndLoadStableXModel,
    "StableXProcessImage": StableXProcessImage,
    "DifferenceExtractorNode": DifferenceExtractorNode

    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadStableXModel": "(Down)load StableX Model",
    "StableXProcessImage": "StableX Process Image",
    "DifferenceExtractorNode": "Extract Difference"
   
    }
