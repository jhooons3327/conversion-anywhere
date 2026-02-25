import os
import sys
import yaml
import torch
import numpy as np
import SimpleITK as sitk
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel
import torch.nn.functional as F
from huggingface_hub import login

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import format_metadata_natural, normalize_ct_slice

# Add MGVQ to path
MGVQ_DIR = Path(__file__).parent.parent / "MGVQ"
if str(MGVQ_DIR) not in sys.path:
    sys.path.insert(0, str(MGVQ_DIR))
from image_embedder import _load_mgvq_model

# --- Plug-and-Play Helpers ---

def get_metadata_parts(metadata):
    """Split metadata into 20 parts for differential attention."""
    ctdivol = metadata.get('CTDIvol', 0.0)
    ctdivol_val = float(ctdivol) if ctdivol is not None else 0.0
    
    return [
        "CT", "scan", "acquired", "on", f"{metadata.get('Manufacturer', 'Unknown')}", 
        "with", f"{metadata.get('ConvolutionKernel', 'Unknown')}", "kernel.\n",
        "Reconstruction", "diameter", f"{metadata.get('ReconstructionDiameter', 0)}mm.\n",
        "Exposure:", f"{metadata.get('ExposureTime', 0)}ms", "at", f"{metadata.get('XRayTubeCurrent', 0)}mA.\n",
        "CTDIvol", f"{ctdivol_val:.2f}mGy.\n",
        "Voxel", "spacing", f"{metadata.get('XYSpacing', 0)}x{metadata.get('ZSpacing', 0)}mm."
    ]

def get_differential_mask(src_meta, tar_meta, weight, device):
    """Compute weight mask for differentiated lines in the prompt."""
    src_parts = get_metadata_parts(src_meta)
    tar_parts = get_metadata_parts(tar_meta)
    
    changed_indices = [i for i in range(20) if src_parts[i] != tar_parts[i]]
    line_mask = torch.ones(5, device=device)
    
    for idx in changed_indices:
        if 0 <= idx <= 7: line_mask[0] = weight
        elif 8 <= idx <= 10: line_mask[1] = weight
        elif 11 <= idx <= 14: line_mask[2] = weight
        elif 15 <= idx <= 16: line_mask[3] = weight
        elif 17 <= idx <= 19: line_mask[4] = weight
        
    return line_mask.view(1, 1, 1, 5)

class PnPFeatureHandler:
    """Handles saving and injecting features for Plug-and-Play inversion."""
    def __init__(self, feature_injection_threshold, self_attn_injection_threshold):
        self.features = {}
        self.mode = "save" # 'save', 'inject', or 'off'
        self.current_timestep = None
        self.feature_injection_threshold = feature_injection_threshold
        self.self_attn_injection_threshold = self_attn_injection_threshold
        self.total_steps = 1
        self.current_step_idx = 0
        self.diff_mask = None
        
    def set_timestep(self, t):
        self.current_timestep = t.item()
        if self.current_timestep not in self.features:
            self.features[self.current_timestep] = {}

    def should_inject_feature(self, idx):
        return (idx / self.total_steps) < self.feature_injection_threshold

    def should_inject_self_attn(self, idx):
        return (idx / self.total_steps) < self.self_attn_injection_threshold

    def save_feature(self, layer, type, tensor):
        if self.mode != 'save': return
        if layer not in self.features[self.current_timestep]:
            self.features[self.current_timestep][layer] = {}
        self.features[self.current_timestep][layer][type] = tensor.detach().cpu() 

    def load_feature(self, layer, type):
        if self.mode != 'inject': return None
        if self.current_timestep in self.features and layer in self.features[self.current_timestep]:
            feat = self.features[self.current_timestep][layer].get(type)
            return feat.to("cuda") if feat is not None else None
        return None

class PnPAttnProcessor:
    """Custom Attention Processor for PnP and Differential Attention."""
    def __init__(self, original_processor, handler, layer_name):
        self.original_processor = original_processor
        self.handler = handler
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        if encoder_hidden_states is not None: # Cross Attention
            batch_size, seq_len, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            query = attn.to_q(hidden_states)
            
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states) if attn.norm_cross else encoder_hidden_states
            key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
            
            query, key, value = map(attn.head_to_batch_dim, (query, key, value))
            probs = attn.get_attention_scores(query, key, attention_mask)
            
            if self.handler.diff_mask is not None:
                mask = self.handler.diff_mask.to(probs.device).squeeze(0)
                probs = probs * mask

            hidden_states = torch.bmm(probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            return attn.to_out[1](attn.to_out[0](hidden_states))

        if self.handler.mode == 'off':
            return self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        # Self Attention Logic
        query, key, value = map(attn.head_to_batch_dim, (attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)))
        
        if self.handler.mode == 'save':
            self.handler.save_feature(self.layer_name, 'q', query)
            self.handler.save_feature(self.layer_name, 'k', key)
        elif self.handler.mode == 'inject' and self.handler.should_inject_self_attn(self.handler.current_step_idx):
            q_saved, k_saved = self.handler.load_feature(self.layer_name, 'q'), self.handler.load_feature(self.layer_name, 'k')
            if q_saved is not None and q_saved.shape == query.shape:
                query, key = q_saved, k_saved

        probs = attn.get_attention_scores(query, key, attn.prepare_attention_mask(attention_mask, hidden_states.shape[1], hidden_states.shape[0]))
        hidden_states = attn.batch_to_head_dim(torch.bmm(probs, value))
        return attn.to_out[1](attn.to_out[0](hidden_states))

def register_pnp_hooks(unet, handler):
    """Register PnP and Differential Attention hooks into UNet."""
    procs = {name: (PnPAttnProcessor(proc, handler, name) if ("attn2" in name or ("up_blocks" in name and "attn1" in name)) else proc)
             for name, proc in unet.attn_processors.items()}
    unet.set_attn_processor(procs)
    
    def spatial_hook(layer):
        def hook(m, i, o):
            if handler.mode == 'save': handler.save_feature(layer, 'spatial', o)
            elif handler.mode == 'inject' and handler.should_inject_feature(handler.current_step_idx):
                saved = handler.load_feature(layer, 'spatial')
                if saved is not None and saved.shape == o.shape: return saved
            return o
        return hook

    if len(unet.up_blocks) > 1:
        for i, resnet in enumerate(unet.up_blocks[1].resnets):
            resnet.register_forward_hook(spatial_hook(f"up_blocks.1.resnets.{i}"))

def sanitize_image(img):
    """Fix invalid metadata in SimpleITK image."""
    for attr in ['Spacing', 'Origin']:
        val = list(getattr(img, f'Get{attr}')())
        if any(np.isnan(v) or (attr == 'Spacing' and v <= 0) for v in val):
            set_val = [1.0 if attr == 'Spacing' else 0.0 for _ in val]
            getattr(img, f'Set{attr}')(tuple(set_val))
    if any(np.isnan(img.GetDirection())):
        img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img

def align_and_process_pair(src_path, tar_path, target_xy=512):
    """Align and resample source and target CT volumes."""
    src_img, tar_img = map(sanitize_image, (sitk.ReadImage(str(src_path)), sitk.ReadImage(str(tar_path))))
    
    def get_bounds(img):
        origin, spacing, size = map(np.array, (img.GetOrigin(), img.GetSpacing(), img.GetSize()))
        return origin, origin + spacing * size, spacing

    s_min, s_max, s_spc = get_bounds(src_img)
    t_min, t_max, t_spc = get_bounds(tar_img)
    i_min, i_max = np.maximum(s_min, t_min), np.minimum(s_max, t_max)
    
    if np.any(i_min >= i_max): i_min, i_max = s_min, s_max
    phys_len = i_max - i_min
    z_size = min(int(phys_len[2]/s_spc[2]), int(phys_len[2]/t_spc[2])) or 1
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(i_min)
    resampler.SetOutputSpacing(phys_len / [target_xy, target_xy, z_size])
    resampler.SetSize([target_xy, target_xy, z_size])
    resampler.SetOutputDirection(src_img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    
    def to_tensor(img):
        arr = sitk.GetArrayFromImage(img)
        return torch.from_numpy(np.stack([np.stack([normalize_ct_slice(s)]*3) for s in arr])).float()

    return to_tensor(resampler.Execute(src_img)), to_tensor(resampler.Execute(tar_img)), src_img, tar_img

def get_text_embeddings(text_model, processor, texts, device):
    """Generate text features for given prompts."""
    processed = [(t.strip().split('\n') + [""]*5)[:5] for t in texts]
    flat = [line for item in processed for line in item]
    with torch.no_grad():
        inputs = processor(text=flat, padding="max_length", return_tensors="pt", truncation=True, max_length=64).to(device)
        return text_model.get_text_features(**inputs).view(len(texts), 5, -1)

@torch.no_grad()
def main(config_path, save_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(open(config_path))
    
    # Load Models
    mgvq = _load_mgvq_model(config['data']['mgvq_model_path'], device)
    processor = AutoProcessor.from_pretrained(config['model']['siglip_model_name'])
    text_model = AutoModel.from_pretrained(config['model']['siglip_model_name']).to(device)
    
    model_cfg = config['model']['unet']
    unet = UNet2DConditionModel(**model_cfg).to(device)
    unet.load_state_dict(torch.load(config['model']['denoising_UNet_weight'], map_location=device)['model_state_dict'])
    unet.eval()

    # Scheduler and PnP Setup
    num_steps, target_t = config['sampling']['num_inference_steps'], config['sampling']['end_t']
    scheduler = DDIMScheduler(num_train_timesteps=config['diffusion']['num_train_timesteps'], beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    steps_arr = np.linspace(1, target_t, num_steps + 1).astype(int)
    seq_timesteps = torch.from_numpy(steps_arr).long().to(device)
    
    pnp_handler = PnPFeatureHandler(config['sampling']['pnp']['feature_injection_threshold'], config['sampling']['pnp']['self_attn_injection_threshold'])
    pnp_handler.total_steps = num_steps
    if config['sampling']['pnp'].get('enable', True): register_pnp_hooks(unet, pnp_handler)
    
    test_pairs = json.load(open(config['data']['test_pairs_json']))
    
    for pair in test_pairs:
        try:
            src_tensor, tar_tensor, src_sitk, _ = align_and_process_pair(pair['src']['path'], pair['tar']['path'])
            slices_lim = config['sampling'].get('total_slices', 16)
            if src_tensor.shape[0] > slices_lim:
                start = (src_tensor.shape[0] - slices_lim) // 2
                src_tensor = src_tensor[start:start+slices_lim]
            
            # Encode and Generate
            latents_0, _, _ = mgvq.encode(src_tensor.to(device))
            src_emb = get_text_embeddings(text_model, processor, [format_metadata_natural(pair['src'])], device).repeat(src_tensor.shape[0], 1, 1)
            tar_emb = get_text_embeddings(text_model, processor, [format_metadata_natural(pair['tar'])], device).repeat(src_tensor.shape[0], 1, 1)
            uncond_emb = get_text_embeddings(text_model, processor, [""], device).repeat(src_tensor.shape[0], 1, 1)
            
            pnp_handler.diff_mask = get_differential_mask(pair['src'], pair['tar'], config['sampling']['differential_attention']['weight'], device) if config['sampling']['differential_attention'].get('enable') else None
            
            # Inversion
            pnp_handler.mode, latents = "save", latents_0.clone()
            for i in range(num_steps):
                t, next_t = seq_timesteps[i], seq_timesteps[i+1]
                pnp_handler.set_timestep(t)
                noise_pred = unet(latents, t, encoder_hidden_states=src_emb).sample
                alpha, n_alpha = scheduler.alphas_cumprod[t], scheduler.alphas_cumprod[next_t]
                latents = (n_alpha**0.5)*((latents - (1-alpha)**0.5*noise_pred)/alpha**0.5) + (1-n_alpha)**0.5*noise_pred

            # Denoising
            pnp_handler.mode, inverted = "inject", latents.clone()
            denoise_steps = seq_timesteps[1:].flip(0)
            for i, t in enumerate(denoise_steps):
                prev_t = denoise_steps[i+1] if i < len(denoise_steps)-1 else torch.tensor(1, device=device)
                pnp_handler.set_timestep(t)
                pnp_handler.current_step_idx = i
                
                n_tar = unet(latents, t, encoder_hidden_states=tar_emb).sample
                if config['sampling'].get('use_neg_cfg'):
                    n_src = unet(latents, t, encoder_hidden_states=src_emb).sample
                    n_pred = n_tar + config['sampling']['guidance_scale']*(n_tar - n_src)
                else:
                    n_uncond = unet(latents, t, encoder_hidden_states=uncond_emb).sample
                    n_pred = n_uncond + config['sampling']['guidance_scale']*(n_tar - n_uncond)
                
                alpha, alpha_prev = scheduler.alphas_cumprod[t], scheduler.alphas_cumprod[prev_t]
                latents = alpha_prev**0.5*((latents-(1-alpha)**0.5*n_pred)/alpha**0.5) + (1-alpha_prev)**0.5*n_pred

            # Decode and Save
            decoded = mgvq.decode(latents).clamp(-1, 1).cpu().numpy()[:, 0]
            out_vol = []
            for idx in range(decoded.shape[0]):
                hu = (decoded[idx]+1)*1000-1000
                out_vol.append(hu.astype(np.int16))
            
            out_img = sitk.GetImageFromArray(np.stack(out_vol))
            out_img.SetSpacing(src_sitk.GetSpacing()); out_img.SetOrigin(src_sitk.GetOrigin()); out_img.SetDirection(src_sitk.GetDirection())
            sitk.WriteImage(out_img, str(Path(save_dir)/f"{pair['tar']['id']}_from_{pair['src']['id']}.nii.gz"))
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_sampling.yml")
    parser.add_argument("--save_dir", type=str, default='results')
    args = parser.parse_args()
    login(token="YOUR_HF_TOKEN")
    main(args.config, args.save_dir)
