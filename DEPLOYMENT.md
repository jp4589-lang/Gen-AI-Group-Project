# Interior Design AI - Pix2Pix with LoRA

FastAPI application for generating edited interior design images using InstructPix2Pix with LoRA fine-tuning.

## Quick Start with Docker

### Prerequisites
- Docker installed
- **Linux with NVIDIA GPU** (recommended for Docker deployment)
- For macOS testing, see "Alternative for macOS Users" section below

### Run with Docker Compose

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-directory>

# Build and run
docker-compose -f docker-compose.interior.yml up --build
```

The API will be available at: **http://localhost:8000**

### Run with Docker (alternative)

```bash
# Build the image
docker build -f Dockerfile.interior -t interior-design-ai .

# Run with GPU support
docker run -p 8000:8000 --gpus all interior-design-ai

# Run with CPU only (slower)
docker run -p 8000:8000 interior-design-ai
```

## 🌐 Access the API

Once the container is running:

- **Interactive Web Interface**: http://localhost:8000
- **API Documentation (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔧 API Endpoints

### 1. Generate Image (JSON Response)
**POST** `/generate`

Upload an interior image and provide an edit instruction to generate a modified version.

**Parameters:**
- `image` (file): Input interior design image
- `prompt` (string): Edit instruction (e.g., "Add a modern chandelier")
- `num_steps` (int, optional): Inference steps [20-100], default: 50
- `guidance_scale` (float, optional): Text guidance [1-15], default: 7.5
- `image_guidance_scale` (float, optional): Image guidance [1-3], default: 1.5
- `resize_to_original` (bool, optional): Resize output to original size, default: true

**Response:** JSON with base64 encoded generated image

### 2. Generate Image (File Download)
**POST** `/generate/file`

Same parameters as above, but returns a PNG file directly for download.

### 3. Health Check
**GET** `/health`

Returns API status and model information.

## 🧪 Testing Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Generate image
curl -X POST http://localhost:8000/generate/file \
  -F "image=@your_room.jpg" \
  -F "prompt=Add a beautiful chandelier to the ceiling" \
  -F "num_steps=50" \
  -o generated.png
```

### Using Python

```python
import requests

# Generate image
with open("room.jpg", "rb") as f:
    files = {"image": f}
    data = {
        "prompt": "Add indoor plants near the window",
        "num_steps": 50
    }
    response = requests.post(
        "http://localhost:8000/generate/file",
        files=files,
        data=data
    )
    
    with open("result.png", "wb") as out:
        out.write(response.content)
```

## 📁 Project Files

- `app.py` - FastAPI application
- `index.html` - Web interface
- `pytorch_lora_weights.safetensors` - Fine-tuned LoRA weights (3.1 MB)
- `requirements_interior.txt` - Python dependencies
- `Dockerfile.interior` - Docker configuration
- `docker-compose.interior.yml` - Docker Compose configuration

## 🤖 Model Information

- **Base Model**: timbrooks/instruct-pix2pix (~1B parameters)
- **LoRA Weights**: ~1.6M parameters (0.15% of base model)
- **Architecture**: Stable Diffusion 1.5 based
- **Training Dataset**: victorzarzu/interior-design-prompt-editing-dataset-train

## ⚙️ Configuration

Default parameters (adjustable via API):
- **Inference Steps**: 50 (balance of quality and speed)
- **Guidance Scale**: 7.5 (text prompt adherence)
- **Image Guidance**: 1.5 (input image preservation)
- **Processing Size**: 512×512 (automatically resized)

## 📊 Performance

- **With GPU (CUDA on Linux):** ~60 seconds per image (50 steps) ✅ Recommended
- **With CPU:** May crash due to memory constraints (Mac Docker limitation)

**⚠️ Important Note for Testing:**

On **macOS**, Docker containers don't have access to Metal (MPS) GPU and run on CPU only, which can cause out-of-memory errors during image generation. The Docker deployment will work perfectly on **Linux systems with NVIDIA GPU support**.

**Alternative for macOS Users (Local Testing):**

If you need to test on Mac, run the FastAPI server locally without Docker:

```bash
# Install dependencies
pip install -r requirements_interior.txt

# Run locally (will use Mac Metal GPU)
python app.py
```

Then access http://localhost:8000 - this will use your Mac's Metal GPU and work properly.

Adjust `num_steps` to balance quality vs speed:
- 20 steps: Fast, lower quality
- 50 steps: Recommended balance
- 100 steps: Best quality, slower

## 🛠️ Troubleshooting

### Container won't start
```bash
# Check container logs
docker-compose -f docker-compose.interior.yml logs -f
```

### GPU not detected
```bash
# Verify NVIDIA drivers
nvidia-smi

# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Out of memory
- Reduce `num_steps` parameter (try 30 instead of 50)
- Use smaller input images
- Use CPU mode if GPU memory is insufficient

## 💡 Example Prompts

- "Add a modern chandelier to the ceiling"
- "Replace the painting with a landscape artwork"
- "Add indoor plants near the window"
- "Change the wall color to light blue"
- "Add a decorative mirror above the sofa"
- "Replace the curtains with white sheer curtains"

## 📄 License

This project uses the InstructPix2Pix model which is subject to its own license terms.

## 🙏 Acknowledgments

- Base model: [InstructPix2Pix by Tim Brooks](https://huggingface.co/timbrooks/instruct-pix2pix)
- Dataset: [Interior Design Dataset by victorzarzu](https://huggingface.co/datasets/victorzarzu/interior-design-prompt-editing-dataset-train)
