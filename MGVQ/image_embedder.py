import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from argparse import Namespace
from pathlib import Path

_mgvq_dir = str(Path(__file__).parent)
if _mgvq_dir not in sys.path:
    sys.path.insert(0, _mgvq_dir)

try:
    from efficientvit.model_zoo import MGVQ_HF
except ImportError as e:
    raise ImportError(f"Ensure dependencies (omegaconf, etc.) are installed. MGVQ dir: {_mgvq_dir}") from e

def _load_mgvq_model(model_path, device):
    """Load MGVQ model from checkpoint."""
    name = os.path.basename(model_path)
    vq_model = "mgvq-f8c32" if "f8c32" in name else "mgvq-f16c32"
    
    model = MGVQ_HF(Namespace(vq_model=vq_model, codebook_groups=8, codebook_size=16384))
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.to(device).eval()

def _normalize_ct_slice(slice_data):
    """Normalize CT slice to [-1, 1]."""
    clipped = np.clip(slice_data, -1024, 1024)
    return ((clipped + 1024.0) / 2048.0 - 0.5) * 2.0

def extract_ct_latent(input_path, model_path, device_str="cuda:0"):
    """Extract latents from NIfTI file or sitk.Image."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = _load_mgvq_model(model_path, device)
    ds_rate = 8 if "f8" in os.path.basename(model_path) else 16

    img = sitk.ReadImage(input_path) if isinstance(input_path, str) else input_path
    size, spc, origin, direction = img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection()
    
    # Resample to 512x512
    new_size = (512, 512, size[2])
    new_spc = (spc[0]*(size[0]/512.0), spc[1]*(size[1]/512.0), spc[2])
    img = sitk.Resample(img, new_size, sitk.Transform(), sitk.sitkLinear, origin, new_spc, direction, 0.0, img.GetPixelID())
    
    data = sitk.GetArrayFromImage(img)
    latents = []
    for z_slice in tqdm(data, desc="Extracting latents"):
        inp = torch.from_numpy(np.stack([_normalize_ct_slice(z_slice)]*3)).unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            quant, _, _ = model.encode(inp)
        latents.append(quant.squeeze(0).permute(1, 2, 0).cpu().numpy())

    latent_vol = np.stack(latents, axis=0).astype(np.float32)
    latent_img = sitk.GetImageFromArray(latent_vol, isVector=True)
    latent_img.SetSpacing((new_spc[0]*ds_rate, new_spc[1]*ds_rate, new_spc[2]))
    latent_img.SetOrigin(origin); latent_img.SetDirection(direction)
    
    return latent_vol, latent_img
