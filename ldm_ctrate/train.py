import os
import glob
import re
import sys
import argparse
import yaml
import json
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from huggingface_hub import login
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset import CTRateIterableDataset

# Add MGVQ to path
MGVQ_DIR = Path(__file__).parent.parent / "MGVQ"
if str(MGVQ_DIR) not in sys.path:
    sys.path.insert(0, str(MGVQ_DIR))

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Replace with your own Hugging Face token
    hf_token = "YOUR_HF_TOKEN"
    login(token=hf_token)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config.yml")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    config = load_config(args.config)
    
    # Setup Logging
    log_dir = os.path.join(os.path.dirname(args.config), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "train.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    checkpoint_dir = config['experiment']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = torch.device(config['training'].get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    
    seed = config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load Models
    print("Loading MedSigLIP...")
    siglip_model_name = config['model']['siglip_model_name']
    processor = AutoProcessor.from_pretrained(siglip_model_name)
    text_model = AutoModel.from_pretrained(siglip_model_name).to(device)
    text_model.eval()

    print("Loading MGVQ...")
    from image_embedder import _load_mgvq_model
    mgvq = _load_mgvq_model(config['data']['mgvq_model_path'], device)
    mgvq.eval()
    
    # Initialize UNet
    model_cfg = config['model']['unet']
    model = UNet2DConditionModel(
        sample_size=model_cfg['sample_size'],
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        layers_per_block=model_cfg.get('layers_per_block', 2),
        block_out_channels=model_cfg['block_out_channels'],
        down_block_types=model_cfg['down_block_types'],
        up_block_types=model_cfg['up_block_types'],
        cross_attention_dim=model_cfg['cross_attention_dim'],
    ).to(device)
    
    train_cfg = config['training']
    optimizer = AdamW(model.parameters(), lr=train_cfg['learning_rate'])

    # Resume Logic
    start_epoch = 0
    resume_path = config['experiment'].get('resume_path', None)
    if config['experiment'].get('resume_from_latest', False):
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt"))
        if ckpt_files:
            resume_path = max(ckpt_files, key=lambda x: int(re.search(r'epoch_(\d+).pt', x).group(1)))

    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    
    # Dataset and Dataloader
    dataset = CTRateIterableDataset(
        config['data']['json_path'], 
        slices_per_kernel=train_cfg.get('slices_per_kernel', 1), 
        shuffle=config['data']['shuffle'], 
        seed=seed
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=config['data']['num_workers'])
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config['diffusion']['num_train_timesteps'],
        prediction_type=config['diffusion'].get('prediction_type', 'epsilon')
    )
    
    # Training Loop
    cfg_prob = train_cfg.get('cfg_prob', 0.2)
    steps_per_epoch = train_cfg.get('steps_per_epoch', 1000)
    iterator = iter(dataloader)
    
    for epoch in range(start_epoch, train_cfg['num_epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        for _ in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
                
            images = batch['image'].to(device)
            texts = batch['text']
            
            # 1. Encode Images to Latents
            with torch.no_grad():
                quant, _, _ = mgvq.encode(images)
                latents = quant * 1.0 
                
            # 2. Encode Text with CFG
            processed_texts = []
            for t in texts:
                if np.random.rand() < cfg_prob:
                    processed_texts.append([""] * 5)
                else:
                    lines = t.strip().split('\n')
                    processed_texts.append((lines + [""] * 5)[:5])
            
            flat_texts = [line for item in processed_texts for line in item]
            with torch.no_grad():
                inputs = processor(text=flat_texts, padding="max_length", return_tensors="pt", truncation=True, max_length=64).to(device)
                text_embeddings = text_model.get_text_features(**inputs).view(len(texts), 5, -1)
                
            # 3. Diffusion Step
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            model_output = model(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            if noise_scheduler.config.prediction_type == "sample":
                loss = F.mse_loss(model_output, latents)
            else:
                loss = F.mse_loss(model_output, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Save Checkpoint
        if (epoch + 1) % train_cfg['save_interval'] == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:04d}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / steps_per_epoch,
                'config': config
            }, ckpt_path)

if __name__ == "__main__":
    main()