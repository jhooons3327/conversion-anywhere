import torch
import numpy as np
import SimpleITK as sitk
import json
import random
from pathlib import Path
from torch.utils.data import IterableDataset

def format_metadata_natural(metadata):
    """Format medical metadata into natural language."""
    ctdivol = metadata.get('CTDIvol', 0.0)
    ctdivol_val = float(ctdivol) if ctdivol is not None else 0.0

    return (
        f"CT scan acquired on {metadata.get('Manufacturer', 'Unknown')} with {metadata.get('ConvolutionKernel', 'Unknown')} kernel.\n"
        f"Reconstruction diameter {metadata.get('ReconstructionDiameter', 0)}mm.\n"
        f"Exposure: {metadata.get('ExposureTime', 0)}ms at {metadata.get('XRayTubeCurrent', 0)}mA.\n"
        f"CTDIvol {ctdivol_val:.2f}mGy.\n"
        f"Voxel spacing {metadata.get('XYSpacing', 0)}x{metadata.get('ZSpacing', 0)}mm."
    )

def normalize_ct_slice(slice_data):
    """Normalize CT slice from HU to [-1, 1]."""
    clipped = np.clip(slice_data, -1000, 1000)
    return (clipped + 1000.0) / 1000.0 - 1.0

class CTRateIterableDataset(IterableDataset):
    def __init__(self, json_path, slices_per_kernel=1, shuffle=True, seed=42):
        super().__init__()
        self.json_path = json_path
        self.slices_per_kernel = slices_per_kernel
        self.shuffle = shuffle
        self.seed = seed
        
        # Target kernels for the study
        self.target_kernels = [
            {'m': 'philips', 'k': 'a'}, {'m': 'philips', 'k': 'b'},
            {'m': 'philips', 'k': 'd'}, {'m': 'philips', 'k': 'ub'},
            {'m': 'pnms', 'k': 'ea'}, {'m': 'pnms', 'k': 'sa'},
            {'m': 'siemens', 'k': "['bl57d', '3']"}, {'m': 'siemens', 'k': "['br36d', '3']"},
            {'m': 'siemens healthineers', 'k': "['bl56f', '3']"},
            {'m': 'siemens healthineers', 'k': "['br40f', '3']"},
            {'m': 'siemens healthineers', 'k': "['br60f', '3']"},
        ]
        
        self._get_key = lambda m, k: f"{m.lower()}_{k.lower()}"
        self.data_buckets = {self._get_key(tk['m'], tk['k']): [] for tk in self.target_kernels}
        
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        for item in raw_data:
            key = self._get_key(item.get('Manufacturer', ''), item.get('ConvolutionKernel', ''))
            if key in self.data_buckets:
                self.data_buckets[key].append(item)

    def _process_volume_slices(self, item, num_slices):
        try:
            sitk_image = sitk.ReadImage(item['path'])
            
            # Robustness fix for metadata
            for attr in ['Spacing', 'Origin']:
                val = list(getattr(sitk_image, f'Get{attr}')())
                if any(np.isnan(v) or (attr == 'Spacing' and v <= 0) for v in val):
                    set_val = [1.0 if attr == 'Spacing' else 0.0 for _ in val]
                    getattr(sitk_image, f'Set{attr}')(tuple(set_val))
            
            # Resample to 512x512
            size = sitk_image.GetSize()
            if size[0] != 512 or size[1] != 512:
                new_size = (512, 512, size[2])
                spc = sitk_image.GetSpacing()
                new_spc = (spc[0]*(size[0]/512.0), spc[1]*(size[1]/512.0), spc[2])
                sitk_image = sitk.Resample(sitk_image, new_size, sitk.Transform(), sitk.sitkLinear, 
                                         sitk_image.GetOrigin(), new_spc, sitk_image.GetDirection(), 0.0, sitk_image.GetPixelID())
            
            arr = sitk.GetArrayFromImage(sitk_image)
            indices = np.random.choice(arr.shape[0], num_slices, replace=(arr.shape[0] < num_slices))
            
            return [{'image': torch.from_numpy(np.stack([normalize_ct_slice(arr[z])]*3)).float(),
                     'text': format_metadata_natural(item)} for z in indices if not np.isnan(arr[z]).any()]
        except Exception as e:
            print(f"Error processing {item.get('path')}: {e}")
            return []

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)
        np.random.seed(seed)
        
        while True:
            batch = []
            for tk in self.target_kernels:
                items = self.data_buckets[self._get_key(tk['m'], tk['k'])]
                if items: batch.extend(self._process_volume_slices(rng.choice(items), self.slices_per_kernel))
            
            if batch:
                yield {
                    'image': torch.stack([s['image'] for s in batch]),
                    'text': [s['text'] for s in batch]
                }
