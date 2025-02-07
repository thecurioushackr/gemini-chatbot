from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetPipeline
)
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
import io
import base64
from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector
import cv2
import logging

logger = logging.getLogger(__name__)

@dataclass
class ImageGenerationConfig:
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: Optional[int] = None
    style_preset: Optional[str] = None
    control_image: Optional[Image.Image] = None
    control_type: Optional[str] = None
    strength: float = 0.8
    init_image: Optional[Image.Image] = None

class AdvancedImageGenerator:
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize with optional cache directory for model weights."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        
        try:
            # Initialize different models
            self.models = {
                "sdxl": StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                    cache_dir=cache_dir
                ).to(self.device),
                
                "flux": DiffusionPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    cache_dir=cache_dir
                ).to(self.device),
                
                "img2img": StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    cache_dir=cache_dir
                ).to(self.device),
                
                "inpaint": StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    cache_dir=cache_dir
                ).to(self.device)
            }

            # Initialize ControlNet models
            self.controlnet_models = {
                "pose": OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=cache_dir),
                "edges": MLSDdetector.from_pretrained("lllyasviel/ControlNet", cache_dir=cache_dir),
                "sketch": HEDdetector.from_pretrained("lllyasviel/ControlNet", cache_dir=cache_dir)
            }

            # Initialize caption generator
            self.caption_processor = AutoProcessor.from_pretrained(
                "microsoft/git-base-coco",
                cache_dir=cache_dir
            )
            self.caption_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/git-base-coco",
                cache_dir=cache_dir
            ).to(self.device)

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

        # Style presets
        self.style_presets = {
            "anime": "anime style, detailed, vibrant colors",
            "realistic": "photorealistic, highly detailed, 8k resolution",
            "artistic": "oil painting style, impressionist, textured",
            "cinematic": "cinematic lighting, dramatic composition, movie still",
            "fantasy": "fantasy art style, magical atmosphere, ethereal lighting",
            "minimalist": "minimalist style, clean lines, simple composition",
            "abstract": "abstract art style, non-representational, geometric shapes",
            "vintage": "vintage style, retro aesthetics, aged appearance",
            "cyberpunk": "cyberpunk style, neon lights, futuristic urban",
            "nature": "natural lighting, organic textures, environmental details"
        }

    def _prepare_prompt(self, config: ImageGenerationConfig) -> str:
        """Enhance prompt with style preset and quality boosters."""
        try:
            base_prompt = config.prompt
            
            # Add style preset if specified
            if config.style_preset and config.style_preset in self.style_presets:
                base_prompt = f"{base_prompt}, {self.style_presets[config.style_preset]}"

            # Add quality boosters
            quality_boosters = "high quality, detailed, sharp focus, professional"
            base_prompt = f"{base_prompt}, {quality_boosters}"

            return base_prompt
        except Exception as e:
            logger.error(f"Error preparing prompt: {e}")
            return config.prompt

    def _prepare_negative_prompt(self, config: ImageGenerationConfig) -> str:
        """Prepare negative prompt with default quality filters."""
        try:
            base_negative = config.negative_prompt or ""
            default_negative = (
                "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                "watermark, signature, text, extra fingers, fewer fingers, "
                "bad hands, bad feet, duplicate, morbid, mutilated"
            )
            
            return f"{base_negative}, {default_negative}" if base_negative else default_negative
        except Exception as e:
            logger.error(f"Error preparing negative prompt: {e}")
            return ""

    def _apply_controlnet(self, image: Image.Image, control_type: str) -> Image.Image:
        """Apply ControlNet preprocessing based on type."""
        try:
            if control_type not in self.controlnet_models:
                raise ValueError(f"Unsupported control type: {control_type}")

            # Convert PIL Image to numpy array
            image_np = np.array(image)

            # Apply appropriate controlnet
            if control_type == "pose":
                control_image = self.controlnet_models["pose"](image_np)
            elif control_type == "edges":
                control_image = self.controlnet_models["edges"](image_np)
            elif control_type == "sketch":
                control_image = self.controlnet_models["sketch"](image_np)

            return Image.fromarray(control_image)
        except Exception as e:
            logger.error(f"Error applying controlnet: {e}")
            return image

    def _generate_caption(self, image: Image.Image) -> str:
        """Generate a caption for the image."""
        try:
            inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
            generated_ids = self.caption_model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=4
            )
            return self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Caption generation failed"

    def generate(self, config: ImageGenerationConfig) -> Dict[str, Any]:
        """Generate images with advanced features and return detailed results."""
        try:
            # Set random seed if specified
            if config.seed is not None:
                torch.manual_seed(config.seed)
                np.random.seed(config.seed)

            # Prepare prompts
            enhanced_prompt = self._prepare_prompt(config)
            negative_prompt = self._prepare_negative_prompt(config)

            # Select appropriate model and generation method
            if config.init_image and config.control_image:
                # Controlled image-to-image generation
                control_image = self._apply_controlnet(config.control_image, config.control_type)
                model = self.models["img2img"]
                images = model(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    image=config.init_image,
                    control_image=control_image,
                    strength=config.strength,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    num_images_per_prompt=config.num_images
                ).images
            elif config.init_image:
                # Standard image-to-image generation
                model = self.models["img2img"]
                images = model(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    image=config.init_image,
                    strength=config.strength,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    num_images_per_prompt=config.num_images
                ).images
            else:
                # Text-to-image generation
                model = self.models["sdxl"] if config.width >= 1024 else self.models["flux"]
                images = model(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    num_images_per_prompt=config.num_images
                ).images

            # Process results
            results = []
            for img in images:
                # Generate caption
                caption = self._generate_caption(img)
                
                # Convert to base64 for easy transmission
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                results.append({
                    "image": img_str,
                    "caption": caption,
                    "metadata": {
                        "prompt": enhanced_prompt,
                        "negative_prompt": negative_prompt,
                        "width": config.width,
                        "height": config.height,
                        "steps": config.num_inference_steps,
                        "guidance_scale": config.guidance_scale,
                        "seed": config.seed,
                        "style_preset": config.style_preset
                    }
                })

            return {
                "status": "success",
                "results": results
            }

        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def interpolate_prompts(self, prompt1: str, prompt2: str, steps: int = 5) -> List[Image.Image]:
        """Generate a sequence of images interpolating between two prompts."""
        try:
            results = []
            for i in range(steps):
                # Calculate interpolation weight
                weight = i / (steps - 1)
                
                # Create interpolated prompt
                interpolated_prompt = f"({prompt1}:{1-weight}) ({prompt2}:{weight})"
                
                # Generate image
                config = ImageGenerationConfig(
                    prompt=interpolated_prompt,
                    num_inference_steps=30  # Reduced steps for interpolation
                )
                result = self.generate(config)
                
                if result["status"] == "success":
                    # Convert base64 back to image
                    img_data = base64.b64decode(result["results"][0]["image"])
                    img = Image.open(io.BytesIO(img_data))
                    results.append(img)

            return results
        except Exception as e:
            logger.error(f"Error in prompt interpolation: {e}")
            return []

    def cleanup(self):
        """Clean up resources and free memory."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Delete models
            del self.models
            del self.controlnet_models
            del self.caption_processor
            del self.caption_model
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
