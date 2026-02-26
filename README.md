# Start Anywhere, Arrive Anywhere: Inversion-Grounded Text-Guided CT Kernel Conversion

Official implementation of the paper: **"Start Anywhere, Arrive Anywhere: Inversion-Grounded Text-Guided CT Kernel Conversion"** (Submitted to MICCAI 2026).

## 🌟 Overview
We introduce **`conversion-anywhere`**, a framework for universal CT kernel conversion. Unlike traditional methods that require paired datasets for specific kernel transitions, our approach leverages Latent Diffusion Models (LDM) and medical metadata to enable conversion between any arbitrary source and target kernels.

**Key Highlights:**
- **Universal Mapping:** "Start Anywhere" (any source kernel) and "Arrive Anywhere" (any target kernel).
- **Inversion-Grounded:** Utilizes DDIM inversion to ensure the output remains strictly faithful to the patient's original anatomy.
- **Text-Guided:** Uses natural language metadata (e.g., "CT scan acquired on Siemens with bl57d kernel") for precise conditioning.

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/anonymous/conversion-anywhere
cd conversion-anywhere

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset: CT-RATE
Our model is trained and validated on the **CT-RATE** dataset, the largest public 3D CT dataset with rich metadata.
- **Download:** [Hugging Face - CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- **Setup:** Place `trainset.json` and `testset.json` in the `ldm_ctrate/` directory. These files should map the `.nii.gz` file paths to their corresponding DICOM metadata.

## 🏗️ External Weights
This project requires two external components:

### 1. MGVQ (Multi-Group VQ) Autoencoder
We use MGVQ for high-fidelity latent mapping. The code is adapted from [MGVQ](https://github.com/MKJia/MGVQ).
- **Model:** `mgvq-f8c32-g8`
- **Weight Download:** [mgvq_f8c32_g8.pt](https://huggingface.co/mkjia/MGVQ/blob/main/mgvq_f8c32_g8.pt)
- **Storage:** Place the file in the `MGVQ/` directory.

### 2. MedSigLIP Encoder
Text embeddings are generated using `google/medsiglip-448`, which will be automatically downloaded during the first run.

---

## 🚀 Training
To train the LDM on your local instance of CT-RATE:

```bash
python ldm_ctrate/train.py --config ldm_ctrate/config_train.yml
```
> **Note on Weights:** Per MICCAI anonymity guidelines and institutional policy, our pre-trained LDM checkpoints are not publicly released. The provided scripts allow for full reproduction of the results by training on the public CT-RATE dataset.

## 🧪 Kernel Conversion (Inference)
To perform kernel conversion, the framework first inverts the source image into the latent space and then re-generates it under the guidance of the target kernel metadata.

1. Set the source image path and target metadata in `ldm_ctrate/config_sampling.yml`.
2. Execute the conversion:

```bash
python ldm_ctrate/sampling.py --config ldm_ctrate/config_sampling.yml
```

### Advanced Sampling Options:
- **Feature Injection:** Control the `pnp` threshold in the config to balance between style (kernel) change and structural preservation.
- **Differential Attention:** Enhances the cross-attention mechanism for better metadata alignment.

## 📂 Project Structure
```text
conversion-anywhere/
├── ldm_ctrate/          # LDM & Diffusion Logic
│   ├── dataset.py       # Metadata-aware data loader
│   ├── train.py         # LDM training entry point
│   ├── sampling.py      # Inversion-grounded conversion script
│   └── unet.py          # Cross-attention UNet architecture
├── MGVQ/                # MGVQ Autoencoder (EfficientViT based)
│   └── ...              # Model and tokenizer components
├── requirements.txt     # Required libraries
└── README.md            # Project documentation
```

## 📜 Acknowledgements
We thank the authors of the following open-source projects:
- [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) for the medical data.
- [MGVQ](https://github.com/MKJia/MGVQ) for the latent space backbone.
- [MedSigLIP](https://huggingface.co/google/medsiglip-448) for the medical vision-language model.

---

### ⚠️ Anonymity Notice
This repository is submitted for double-blind review at MICCAI 2026. All author information, affiliations, and personal identifiers have been removed. Any mention of "our" or "this work" refers to the anonymous authors of the submission.
