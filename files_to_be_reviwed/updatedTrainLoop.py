import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from stegwatermark.models import Encoder, Decoder, Discriminator
from stegwatermark.data import WatermarkDataset
from stegwatermark.distort import apply_distortions

class TrainConfig:
    def __init__(self, data_dir, save_dir, resume="", synthetic=False, batch_size=4, num_steps=140000, device="cuda"):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.resume = resume
        self.synthetic = synthetic
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.device = device
        self.save_every = 5000
        self.log_every = 100

def run_training(config: TrainConfig):
    device = torch.device(config.device)
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Initialize Core StegaStamp Framework
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    critic = Discriminator().to(device)
    
    # Load dataset structure
    dataset = WatermarkDataset(config.data_dir, size=400)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)
    data_iter = iter(dataloader)
    
    print(f"covers: {len(dataset)}")
    
    # Optimizers
    opt_g = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    opt_d = torch.optim.RMSprop(critic.parameters(), lr=1e-5)
    
    # Initialize Perceptual Metrics
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    
    start_step = 0
    
    # Load Checkpoint Safe Recovery Frame
    if config.resume and os.path.exists(config.resume):
        print(f"Loading checkpoint weights from: {config.resume}")
        checkpoint = torch.load(config.resume, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        critic.load_state_dict(checkpoint['critic'])
        opt_g.load_state_dict(checkpoint['opt_g'])
        opt_d.load_state_dict(checkpoint['opt_d'])
        start_step = checkpoint['step']
        print(f"resumed at step {start_step + 1}")
    
    # Training Loop Execution
    for step in range(start_step, config.num_steps):
        encoder.train()
        decoder.train()
        critic.train()
        
        # Pull next batch of clean cover images
        try:
            covers, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            covers, _ = next(data_iter)
            
        covers = covers.to(device)
        
        # Generate target random bit signatures (Secret Message payload)
        secret_bits = torch.randint(0, 2, (config.batch_size, 100), dtype=torch.float32).to(device)
        
        # --- RECOVERY & "KICK" LOSS SCHEDULER BLOCK ---
        # If the model is recovering right past step 30,000 out of a stagnant zone,
        # aggressively tilt weights toward finding the message bit pattern first.
        if 30000 <= step < 38000:
            secret_weight = 5.0     # Boosted message priority
            yuv_weight = 0.05       # Slashed strict color penalties
            lpips_weight = 0.05     # Slashed structure metrics
            crop_factor = 0.10
        else:
            # Baseline StegaStamp schedules
            secret_weight = 1.0
            yuv_weight = 2.0 if step >= 1500 else 0.0
            lpips_weight = 1.5 if step >= 1500 else 0.0
            crop_factor = min(0.10, (step - 1500) / 100000.0) if step >= 1500 else 0.0
            
        # 1. Forward Pass: Encode image
        stamped_images = encoder(covers, secret_bits)
        
        # 2. Apply Spatial Distortions and Edge Crops
        distorted_images = apply_distortions(stamped_images, crop_val=crop_factor)
        
        # 3. Decode Message Payload
        predicted_bits = decoder(distorted_images)
        clean_predicted_bits = decoder(stamped_images)
        
        # 4. Compute Loss Formulations
        loss_secret = F.binary_cross_entropy_with_logits(predicted_bits, secret_bits)
        
        # Image preservation losses
        loss_yuv = F.mse_loss(stamped_images, covers)
        loss_lpips = lpips_fn(stamped_images, covers).mean()
        
        # Critic/GAN Loss Component
        g_loss_adv = -torch.mean(critic(stamped_images))
        
        # Global Generator Optimization Goal
        total_g_loss = (secret_weight * loss_secret) + (yuv_weight * loss_yuv) + (lpips_weight * loss_lpips) + (0.5 * g_loss_adv)
        
        opt_g.zero_grad()
        total_g_loss.backward()
        opt_g.step()
        
        # 5. Critic Optimization Step
        opt_d.zero_grad()
        d_loss = torch.mean(critic(stamped_images.detach())) - torch.mean(critic(covers))
        d_loss.backward()
        opt_d.step()
        
        # Calculate Validation Accuracies
        with torch.no_grad():
            bit_acc = ((predicted_bits > 0).float() == secret_bits).float().mean().item()
            clean_acc = ((clean_predicted_bits > 0).float() == secret_bits).float().mean().item()
            
        # Logging Outputs
        if step % config.log_every == 0 or step == start_step:
            print(f"step {step}  bit_acc {bit_acc:.3f}  clean {clean_acc:.3f}  secret {loss_secret:.3f}  yuv {loss_yuv:.3f}  lpips {loss_lpips:.3f}  crop {crop_factor:.3f}")
            
        # Automatic Progress Checkpoints Saving
        if (step > 0 and step % config.save_every == 0) or step == config.num_steps - 1:
            save_path = os.path.join(config.save_dir, f"step_{step:06d}.pt")
            latest_path = os.path.join(config.save_dir, "latest.pt")
            
            state = {
                'step': step,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'critic': critic.state_dict(),
                'opt_g': opt_g.state_dict(),
                'opt_d': opt_d.state_dict()
            }
            torch.save(state, save_path)
            torch.save(state, latest_path)
            print(f"Saved secure checkpoint: {save_path}")
