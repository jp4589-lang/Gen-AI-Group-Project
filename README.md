# Interior Design AI - InstructPix2Pix with LoRA

**Generative AI Course - Final Project**

An AI-powered interior design image editor using InstructPix2Pix fine-tuned with LoRA (Low-Rank Adaptation). Generate realistic interior design modifications through natural language instructions.

## 👥 Team

Sachin Otsuka Arjun (so2748), Sze Ning Tan (st3730), Jui-Chen Pu (jp4589), Charlotte Niklahs (ccn2128), Keshav Ramesh (kr3150), Aidynbek Mussa (am6565)

## 🎯 Project Overview

This project implements a production-ready FastAPI application that allows users to edit interior design images using text prompts. The model is based on InstructPix2Pix (~1B parameters) fine-tuned with LoRA weights (~1.6M parameters) on the interior design dataset.

### Key Features

- **FastAPI REST API** with automatic documentation
- **Beautiful web interface** with drag-and-drop upload
- **Docker deployment** ready for production
- **LoRA fine-tuning** for efficient model adaptation (0.15% additional parameters)
- **GPU acceleration** support (CUDA/MPS)
- **Interactive API documentation** (Swagger UI)

## 🚀 Quick Start

### Prerequisites

- **Docker** (recommended for deployment)
- **Linux with NVIDIA GPU** (recommended for best performance)
- **Python 3.10+** (for local development)

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/jp4589-lang/Gen-AI-Group-Project.git
cd Gen-AI-Group-Project

# Build and run with Docker Compose
docker-compose -f docker-compose.interior.yml up --build
```

The API will be available at **http://localhost:8000**

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements_interior.txt

# Run the FastAPI server
python app.py
```

Access the application at **http://localhost:8000**

## 📂 Project Structure

```
.
├── app.py                          # FastAPI application (main code)
├── index.html                      # Web interface
├── pytorch_lora_weights.safetensors # Trained LoRA weights (3.1 MB)
├── requirements_interior.txt       # Python dependencies
├── Dockerfile.interior             # Docker configuration
├── docker-compose.interior.yml     # Docker Compose setup
├── DEPLOYMENT.md                   # Detailed deployment guide
├── .dockerignore                   # Docker build exclusions
└── README.md                       # This file
```

## 🌐 API Endpoints

### Web Interface
- **GET /** - Interactive web interface for image generation

### API Documentation
- **GET /docs** - Swagger UI (interactive API documentation)
- **GET /redoc** - ReDoc alternative documentation

### Health Check
- **GET /health** - Check API status and model information

### Image Generation
- **POST /generate** - Generate edited image (JSON with base64)
- **POST /generate/file** - Generate edited image (direct PNG download)

**Parameters:**
- `image` (file, required): Input interior image
- `prompt` (string, required): Edit instruction (e.g., "Add a modern chandelier")
- `num_steps` (int, optional): Inference steps [20-100], default: 50
- `guidance_scale` (float, optional): Text guidance [1-15], default: 7.5
- `image_guidance_scale` (float, optional): Image guidance [1-3], default: 1.5
- `resize_to_original` (bool, optional): Resize output to match input, default: true

## 💡 Example Usage

### Using the Web Interface

1. Open http://localhost:8000
2. Drag and drop an interior image
3. Enter a prompt (e.g., "Add indoor plants near the window")
4. Click "Generate" and wait for results

### Using cURL

```bash
# Generate image and save as file
curl -X POST http://localhost:8000/generate/file \
  -F "image=@room.jpg" \
  -F "prompt=Add a beautiful chandelier to the ceiling" \
  -F "num_steps=50" \
  -o generated.png
```

### Using Python

```python
import requests

with open("room.jpg", "rb") as f:
    files = {"image": f}
    data = {
        "prompt": "Replace the painting with a landscape artwork",
        "num_steps": 50,
        "guidance_scale": 7.5
    }
    response = requests.post(
        "http://localhost:8000/generate/file",
        files=files,
        data=data
    )
    
    with open("result.png", "wb") as out:
        out.write(response.content)
```

## 🤖 Model Information

### Base Model
- **Name:** timbrooks/instruct-pix2pix
- **Architecture:** Stable Diffusion 1.5 based
- **Parameters:** ~1 billion
- **Components:**
  - VAE (encoder/decoder): Image ↔ Latent space conversion
  - CLIP Text Encoder: Natural language understanding
  - UNet: Denoising diffusion model

### LoRA Fine-Tuning
- **Parameters:** ~1.6 million (0.15% of base model)
- **Dataset:** victorzarzu/interior-design-prompt-editing-dataset-train
- **Training Split:** 80/20 train/test (4259 examples)
- **File Size:** 3.1 MB

### Training Details
- **Method:** LoRA (Low-Rank Adaptation)
- **Timesteps:** 1000 (training), 50 (inference default)
- **Processing Size:** 512×512 latent space with 8x compression
- **Output:** Automatically resized to original image dimensions
 
### Updated Training Workflow (Local Dataset + Custom Training Script)

This project uses a modified LoRA training pipeline adapted from the Diffusers InstructPix2Pix implementation. We do not train using the original HuggingFace dataset format directly. Instead, we generate a local disk-backed Arrow dataset using `split_dataset.py` and then train using our updated training script located inside the Diffusers repository under:

```
diffusers/examples/research_projects/instructpix2pix_lora/train_instruct_pix2pix_lora.py
```

This script is the version we edited for our use case and fully replaces the default Diffusers script.

---

### 1. Prepare the Dataset Locally (Run First)

We first download the dataset from HuggingFace and convert it into a local disk-based Arrow dataset that Diffusers can train on.

Run:

```bash
python split_dataset.py
```

This script performs:

1. Downloads the dataset from:  
   `victorzarzu/interior-design-prompt-editing-dataset-train`
2. Splits it using an **80% train / 20% test** ratio  
3. Saves it into a folder with the following structure:

```
interior_design_split/
    train/
        data-00000-of-00003.arrow
        data-00001-of-00003.arrow
        data-00002-of-00003.arrow
        dataset_info.json
        state.json
    test/
        data-00000-of-00001.arrow
        dataset_info.json
        state.json
    dataset_dict.json
```

This folder is what you point to using the `--dataset_name` argument in the training command.

---

### 2. Replace Diffusers’ Original Training Script

Clone Diffusers (optional, only if you want the reference structure):

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers/examples/research_projects/instructpix2pix_lora
pip install -r requirements.txt
pip install accelerate
```

Place **our updated training script** into the same directory above, replacing the original:

```
train_instruct_pix2pix_lora.py
```

The updated script contains:

- Correct column bindings for our dataset:  
  `original_image`, `designed_image`, `edit_prompt`
- Loading the dataset using `load_from_disk` instead of `load_dataset`
- Stable logging and configuration controls
- Saving LoRA adapters to:  
  `ip2p-lora-interior/`

---

### 3. Launch Training (Using Our Updated Script)

After running `split_dataset.py` and replacing the training script, launch training:

```bash
accelerate launch train_instruct_pix2pix_lora.py \
  --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
  --dataset_name="Your_Dataset_Path" \ 
  --original_image_column="original_image" \
  --edited_image_column="designed_image" \
  --edit_prompt_column="edit_prompt" \
  --resolution=256 \
  --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=20 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="ip2p-lora-interior" \
  --seed=42
```

This creates a trained LoRA adapter folder:

```
ip2p-lora-interior/
    adapter_config.json
    pytorch_lora_weights.safetensors
```

These weights are the same ones loaded by the FastAPI backend:

```
pytorch_lora_weights.safetensors
```

---

### Summary

The final training workflow:

1. Run `split_dataset.py` to generate `interior_design_split/`
2. Replace Diffusers’ training script with our customized version
3. Launch training via `accelerate`
4. Use the exported LoRA weights inside the FastAPI application

This ensures reproducible fine-tuning and consistent formatting aligned with our dataset and application requirements.

## 📊 Performance

| Environment | Time per Image (50 steps) | Status |
|------------|---------------------------|---------|
| NVIDIA GPU (CUDA) | ~60 seconds | ✅ Recommended |
| Mac Metal (MPS) | ~60 seconds | ✅ Supported (local only) |
| CPU | 5-10 minutes | ⚠️ Slow, may crash in Docker |

### Performance Tips

- **Adjust steps:** Lower steps (20-30) for faster generation, higher (70-100) for better quality
- **Use GPU:** Significantly faster and more memory-efficient
- **Batch processing:** Use the API endpoints for multiple images

## ⚠️ Important Notes

### Docker on macOS Limitation

Docker containers on macOS **cannot access Metal (MPS) GPU** and run on CPU only, which may cause out-of-memory errors during inference. This is a platform limitation, not a code issue.

**Solution for Mac users:**
Run the FastAPI server locally without Docker to use Metal GPU:
```bash
pip install -r requirements_interior.txt
python app.py
```

**For deployment:** The Docker setup works perfectly on **Linux with NVIDIA GPU**, which is the recommended deployment environment.

## 🧪 Example Prompts

Good prompts for interior design editing:
- "Add a modern chandelier to the ceiling"
- "Replace the painting with a landscape artwork"
- "Add indoor plants near the window"
- "Change the wall color to light blue"
- "Add a decorative mirror above the sofa"
- "Replace the curtains with white sheer curtains"
- "Add pendant lights above the dining table"

## 🛠️ Troubleshooting

### Model not loading
- Ensure `pytorch_lora_weights.safetensors` exists in the project directory
- Check available disk space (model cache requires ~5GB)

### Container crashes during generation
- Increase Docker memory allocation
- Use Linux with NVIDIA GPU instead of macOS
- Run locally without Docker on Mac

### Out of memory errors
- Reduce `num_steps` parameter (try 30 instead of 50)
- Use smaller input images
- Ensure adequate system memory

### GPU not detected in Docker
```bash
# Check NVIDIA drivers
nvidia-smi

# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 📖 Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Comprehensive deployment guide
- **API Docs:** http://localhost:8000/docs (when running)
- **Health Check:** http://localhost:8000/health

## 📄 License

This project uses the InstructPix2Pix model which is subject to its own license terms.

## 🙏 Acknowledgments

- **Base Model:** [InstructPix2Pix by Tim Brooks](https://huggingface.co/timbrooks/instruct-pix2pix)
- **Dataset:** [Interior Design Dataset by victorzarzu](https://huggingface.co/datasets/victorzarzu/interior-design-prompt-editing-dataset-train)
- **Framework:** FastAPI, Diffusers, PyTorch, Transformers
