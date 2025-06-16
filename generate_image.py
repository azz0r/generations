#!/usr/bin/env python3
"""
Image Generation Script using Hugging Face Diffusers
Generates images from text prompts using Stable Diffusion
"""

import torch
from diffusers import StableDiffusionPipeline
import argparse
import os
from datetime import datetime

def generate_image(prompt, output_dir="output", model_id="runwayml/stable-diffusion-v1-5"):
    """
    Generate an image from a text prompt
    
    Args:
        prompt (str): Text prompt for image generation
        output_dir (str): Directory to save generated images
        model_id (str): Hugging Face model identifier
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model: {model_id}")
    print("This may take a while on first run as the model downloads...")
    
    # Load the pipeline
    # Use CPU for compatibility, change to "cuda" if you have GPU
    device = "cpu"
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for CPU
        safety_checker=None,  # Disable safety checker for faster inference
        requires_safety_checker=False
    )
    
    pipeline = pipeline.to(device)
    
    print(f"Generating image for prompt: '{prompt}'")
    
    # Generate image
    with torch.no_grad():
        image = pipeline(
            prompt,
            num_inference_steps=20,  # Reduce steps for faster generation
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
    
    # Save image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    image.save(filepath)
    print(f"Image saved to: {filepath}")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--output-dir", default="output", help="Output directory for images")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Model to use")
    
    args = parser.parse_args()
    
    try:
        filepath = generate_image(args.prompt, args.output_dir, args.model)
        print(f"✅ Successfully generated image: {filepath}")
    except Exception as e:
        print(f"❌ Error generating image: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())