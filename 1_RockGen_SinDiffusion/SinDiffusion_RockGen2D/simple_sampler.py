"""
Simple sampler for trained SinDiffusion models using .pth checkpoints.
This script works with the checkpoints saved by proper_sindiffusion_train.py.
"""

import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

# Import the original model components
from guided_diffusion.sinddpm import UNetModel
from guided_diffusion.script_util import create_gaussian_diffusion


def main():
    parser = argparse.ArgumentParser(description='Simple SinDiffusion Sampler for .pth checkpoints')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to input image (for reference)')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='RESULT/simple_sampling', help='Output directory for generated images')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Simple SinDiffusion Sampler")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Input image: {args.data_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Input image not found at {args.data_dir}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    print("Loading trained model...")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        
        # Extract model parameters from checkpoint
        model_args = checkpoint['args']
        print(f"Model was trained with: {model_args}")
        
        # Create model with same architecture
        model = UNetModel(
            image_size=model_args.image_size,
            in_channels=3,
            model_channels=model_args.num_channels,
            out_channels=3,
            num_res_blocks=model_args.num_res_blocks,
            attention_resolutions=tuple(int(x) for x in model_args.attention_resolutions.split(",")),
            dropout=0.0,
            channel_mult=tuple(int(x) for x in model_args.channel_mult.split(",")),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=model_args.use_checkpoint,
            use_fp16=model_args.use_fp16,
            num_heads=model_args.num_heads,
            num_head_channels=model_args.num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=model_args.use_scale_shift_norm,
            resblock_updown=model_args.resblock_updown,
            use_new_attention_order=False,
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create diffusion process
    print("Creating diffusion process...")
    diffusion = create_gaussian_diffusion(
        steps=model_args.diffusion_steps,
        learn_sigma=False,
        sigma_small=True,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    generated_images = []
    
    with torch.no_grad():
        for batch_idx in range(0, args.num_samples, args.batch_size):
            batch_size = min(args.batch_size, args.num_samples - batch_idx)
            print(f"Generating batch {batch_idx//args.batch_size + 1}/{(args.num_samples + args.batch_size - 1)//args.batch_size}")
            
            # Generate sample using diffusion
            sample = diffusion.p_sample_loop(
                model=model,
                shape=(batch_size, 3, model_args.image_size, model_args.image_size),
                noise=None,
                clip_denoised=True,
                denoised_fn=None,
                model_kwargs={},
                device=device,
                progress=False
            )
            
            # Denormalize
            generated_image = (sample + 1) / 2  # Denormalize from [-1,1] to [0,1]
            generated_image = torch.clamp(generated_image, 0, 1)
            
            # Save individual samples
            for i in range(batch_size):
                sample_idx = batch_idx + i + 1
                if sample_idx > args.num_samples:
                    break
                    
                from torchvision.utils import save_image
                sample_path = os.path.join(args.output_dir, f'generated_sample_{sample_idx:02d}.png')
                save_image(generated_image[i], sample_path)
                print(f"Saved: {sample_path}")
                
                generated_images.append(generated_image[i])
    
    print(f"\n✅ Generation complete! Generated {args.num_samples} samples.")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
