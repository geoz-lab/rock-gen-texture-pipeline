"""
Proper SinDiffusion training using the original model architecture.
This removes MPI dependencies but keeps the real diffusion process.
"""

import os
import argparse
import time
import math
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

# Import the original model components
from guided_diffusion.sinddpm import UNetModel
from guided_diffusion.script_util import create_gaussian_diffusion
from guided_diffusion.nn import mean_flat


def main():
    parser = argparse.ArgumentParser(description='Proper SinDiffusion Training (Original Architecture)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to input image')
    parser.add_argument('--image_size', type=int, default=256, help='Training image size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--save_interval', type=int, default=100, help='Save model every N epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='Log every N epochs')
    parser.add_argument('--num_channels', type=int, default=64, help='Number of channels')
    parser.add_argument('--num_res_blocks', type=int, default=1, help='Number of residual blocks')
    parser.add_argument('--channel_mult', type=str, default="1,2,4", help='Channel multipliers')
    parser.add_argument('--attention_resolutions', type=str, default="2", help='Attention resolutions')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_head_channels', type=int, default=16, help='Number of channels per head')
    parser.add_argument('--use_scale_shift_norm', action='store_true', help='Use scale shift norm')
    parser.add_argument('--resblock_updown', action='store_true', help='Use residual blocks for up/downsampling')
    parser.add_argument('--use_fp16', action='store_true', help='Use FP16')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use gradient checkpointing')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Proper SinDiffusion Training (Original Architecture)")
    print("=" * 60)
    print(f"Input image: {args.data_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Training epochs: {args.epochs}")
    print(f"Num channels: {args.num_channels}")
    print(f"Num res blocks: {args.num_res_blocks}")
    print(f"Channel mult: {args.channel_mult}")
    print(f"Attention resolutions: {args.attention_resolutions}")
    print(f"Num heads: {args.num_heads}")
    print(f"Num head channels: {args.num_head_channels}")
    print(f"Use scale shift norm: {args.use_scale_shift_norm}")
    print(f"Resblock updown: {args.resblock_updown}")
    print(f"Use FP16: {args.use_fp16}")
    print(f"Use checkpoint: {args.use_checkpoint}")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Image not found at {args.data_dir}")
        return
    
    # Setup GPU
    device = setup_gpu()
    
    # Load and preprocess image
    print("Loading and preprocessing image...")
    image = Image.open(args.data_dir).convert('RGB')
    
    # Resize image
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(image)
    print(f"Image loaded: {image_tensor.shape}")
    
    # Parse channel multipliers
    channel_mult = tuple(int(x) for x in args.channel_mult.split(","))
    attention_resolutions = tuple(int(x) for x in args.attention_resolutions.split(","))
    
    # Create the original UNet model
    print("Creating original UNet model...")
    model = UNetModel(
        image_size=args.image_size,
        in_channels=3,
        model_channels=args.num_channels,
        out_channels=3,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=0.0,
        channel_mult=channel_mult,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=args.use_checkpoint,
        use_fp16=args.use_fp16,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=-1,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown,
        use_new_attention_order=False,
    )
    
    # Create the diffusion process using the proper function
    print("Creating diffusion process...")
    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=False,
        sigma_small=True,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )
    
    # Setup training
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Create target (same as input for single image training)
    target = image_tensor.unsqueeze(0).repeat(args.batch_size, 1, 1, 1)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("Starting proper diffusion training...")
    start_time = time.time()
    losses = []
    
    for epoch in range(args.epochs):
        try:
            model.train()
            optimizer.zero_grad()
            
            # PROPER DIFFUSION TRAINING using original diffusion process
            # 1. Sample random timesteps
            t = torch.randint(0, args.diffusion_steps, (args.batch_size,), device=device)
            
            # 2. Use the original diffusion training losses
            losses_dict = diffusion.training_losses(
                model=model,
                x_start=target,
                t=t,
                model_kwargs={},
                noise=None
            )
            
            loss = losses_dict["loss"].mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Record loss
            losses.append(loss.item())
            
            # Logging
            if epoch % args.log_interval == 0:
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:4d}/{args.epochs}, Loss: {loss.item():.6f}, LR: {lr:.2e}, Time: {elapsed:.1f}s")
            
            # Save model
            if epoch % args.save_interval == 0 and epoch > 0:
                save_model(model, optimizer, epoch, args, device)
            
            # Clear GPU cache periodically
            if epoch % 25 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error at epoch {epoch}: {e}")
                print("Attempting to recover...")
                torch.cuda.empty_cache()
                time.sleep(1)
                continue
            else:
                raise e
    
    print("Training completed!")
    
    # Final save
    save_model(model, optimizer, args.epochs, args, device)
    
    # Test generation with proper diffusion sampling
    test_diffusion_generation(model, diffusion, image_tensor, args, device)


def setup_gpu():
    """Setup GPU with error handling."""
    if torch.cuda.is_available():
        try:
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Set CUDA environment variables for stability
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['CUDA_CACHE_DISABLE'] = '1'
            
            # Test GPU with simple operation
            test_tensor = torch.randn(1, 1, 64, 64).cuda()
            test_output = torch.nn.functional.conv2d(test_tensor, torch.randn(1, 1, 3, 3).cuda())
            print("GPU test successful!")
            
            torch.cuda.empty_cache()
            return device
            
        except Exception as e:
            print(f"GPU setup failed: {e}")
            print("Falling back to CPU...")
            return torch.device('cpu')
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')


def save_model(model, optimizer, epoch, args, device):
    """Save model checkpoint."""
    output_dir = f"OUTPUT/proper_sindiffusion_{os.path.basename(args.data_dir).split('.')[0]}"
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    
    checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")


def test_diffusion_generation(model, diffusion, original_image, args, device):
    """Test model generation with proper diffusion sampling."""
    print("Testing proper diffusion generation...")
    model.eval()
    
    with torch.no_grad():
        # Use the original diffusion sampling process
        sample = diffusion.p_sample_loop(
            model=model,
            shape=(1, 3, args.image_size, args.image_size),
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs={},
            device=device,
            progress=True
        )
        
        # Denormalize and save
        generated_image = (sample + 1) / 2  # Denormalize from [-1,1] to [0,1]
        generated_image = torch.clamp(generated_image, 0, 1)
        
        # Save generated sample
        output_dir = f"OUTPUT/proper_sindiffusion_{os.path.basename(args.data_dir).split('.')[0]}"
        os.makedirs(output_dir, exist_ok=True)
        
        from torchvision.utils import save_image
        save_image(generated_image[0], os.path.join(output_dir, 'proper_generated_sample.png'))
        print(f"Proper diffusion sample saved to: {output_dir}")


if __name__ == "__main__":
    main()
