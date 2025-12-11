"""
Interior Design AI - FastAPI Application
Pix2Pix with LoRA fine-tuning for interior design image editing
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import io
import base64
import warnings
import os
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI(
    title="Interior Design AI - Pix2Pix with LoRA",
    description="Generate edited interior design images using InstructPix2Pix with LoRA fine-tuning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the model
pipe = None
device = None

# Configuration
LORA_WEIGHTS_PATH = Path(__file__).parent / "pytorch_lora_weights.safetensors"
BASE_MODEL_ID = "timbrooks/instruct-pix2pix"


@app.on_event("startup")
async def startup_event():
    """Load the InstructPix2Pix model with LoRA weights on startup"""
    global pipe, device
    
    print("\n" + "="*60)
    print("🏠 Interior Design AI - Pix2Pix with LoRA")
    print("="*60)
    print("⏳ Loading InstructPix2Pix model...")
    
    # Load base model
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    # Load LoRA weights if available
    if LORA_WEIGHTS_PATH.exists():
        print(f"⏳ Loading LoRA weights from {LORA_WEIGHTS_PATH}...")
        try:
            pipe.load_lora_weights(LORA_WEIGHTS_PATH.parent, weight_name=LORA_WEIGHTS_PATH.name)
            print("✅ LoRA weights loaded successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Could not load LoRA weights: {e}")
            print("   Running with base InstructPix2Pix model only")
    else:
        print(f"⚠️  LoRA weights not found at {LORA_WEIGHTS_PATH}")
        print("   Running with base InstructPix2Pix model only")
    
    # Determine device (MPS for Mac, CUDA for NVIDIA, CPU fallback)
    if torch.backends.mps.is_available():
        device = "mps"
        pipe.to(device)
        print("🚀 Using Metal (MPS) GPU")
    elif torch.cuda.is_available():
        device = "cuda"
        pipe.to(device)
        print("🚀 Using CUDA GPU")
    else:
        device = "cpu"
        pipe.to(torch.float32)
        print("⚠️  Using CPU (this will be slower)")
    
    print("✅ Model loaded and ready!")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """Root endpoint - serve web interface if available, otherwise return API info"""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    return {
        "message": "Interior Design AI - Pix2Pix with LoRA",
        "description": "API for generating edited interior design images",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "generate_json": "/generate",
            "generate_file": "/generate/file"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and model status"""
    return {
        "status": "healthy" if pipe is not None else "unhealthy",
        "model_loaded": pipe is not None,
        "device": device,
        "model": BASE_MODEL_ID,
        "lora_weights": str(LORA_WEIGHTS_PATH) if LORA_WEIGHTS_PATH.exists() else "not found"
    }


@app.post("/generate")
async def generate_image(
    image: UploadFile = File(..., description="Input interior design image"),
    prompt: str = Form(..., description="Edit instruction (e.g., 'Add a modern chandelier')"),
    num_steps: int = Form(50, ge=20, le=100, description="Number of inference steps (20-100)"),
    guidance_scale: float = Form(7.5, ge=1.0, le=15.0, description="Text guidance scale (1-15)"),
    image_guidance_scale: float = Form(1.5, ge=1.0, le=3.0, description="Image guidance scale (1-3)"),
    resize_to_original: bool = Form(True, description="Resize output to original image dimensions")
):
    """
    Generate edited interior design image (JSON response with base64 encoded image)
    
    Args:
        image: Input interior image file
        prompt: Edit instruction describing the desired change
        num_steps: Number of diffusion steps (more steps = better quality, slower)
        guidance_scale: How closely to follow the text prompt
        image_guidance_scale: How closely to follow the input image
        resize_to_original: Whether to resize output to match input dimensions
    
    Returns:
        JSON with base64 encoded generated image and metadata
    """
    try:
        # Validate model is loaded
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
        
        # Read and validate input image
        contents = await image.read()
        try:
            input_image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        original_size = input_image.size
        
        # Resize to 512x512 for model processing
        input_image_resized = input_image.resize((512, 512), Image.LANCZOS)
        
        print(f"\n🎨 Generating edit: '{prompt}'")
        print(f"   Parameters: steps={num_steps}, guidance={guidance_scale}, img_guidance={image_guidance_scale}")
        print(f"   Original size: {original_size}, Processing size: (512, 512)")
        
        # Generate the edited image
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=input_image_resized,
                num_inference_steps=num_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale
            )
        
        generated_image = result.images[0]
        
        # Resize back to original dimensions if requested
        output_size = original_size if resize_to_original else (512, 512)
        if resize_to_original and output_size != (512, 512):
            generated_image = generated_image.resize(output_size, Image.LANCZOS)
            print(f"   Resized output to: {output_size}")
        
        # Convert to base64 for JSON response
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        print("✅ Generation complete!\n")
        
        return {
            "success": True,
            "prompt": prompt,
            "image": img_base64,
            "original_size": list(original_size),
            "output_size": list(output_size),
            "parameters": {
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "image_guidance_scale": image_guidance_scale,
                "resize_to_original": resize_to_original
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/file")
async def generate_image_file(
    image: UploadFile = File(..., description="Input interior design image"),
    prompt: str = Form(..., description="Edit instruction (e.g., 'Add a modern chandelier')"),
    num_steps: int = Form(50, ge=20, le=100, description="Number of inference steps (20-100)"),
    guidance_scale: float = Form(7.5, ge=1.0, le=15.0, description="Text guidance scale (1-15)"),
    image_guidance_scale: float = Form(1.5, ge=1.0, le=3.0, description="Image guidance scale (1-3)"),
    resize_to_original: bool = Form(True, description="Resize output to original image dimensions")
):
    """
    Generate edited interior design image (direct PNG file download)
    
    Same parameters as /generate endpoint, but returns the image file directly
    instead of JSON with base64 encoding.
    """
    try:
        # Validate model is loaded
        if pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
        
        # Read and validate input image
        contents = await image.read()
        try:
            input_image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        original_size = input_image.size
        
        # Resize to 512x512 for model processing
        input_image_resized = input_image.resize((512, 512), Image.LANCZOS)
        
        print(f"\n🎨 Generating edit: '{prompt}'")
        print(f"   Parameters: steps={num_steps}, guidance={guidance_scale}, img_guidance={image_guidance_scale}")
        
        # Generate the edited image
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                image=input_image_resized,
                num_inference_steps=num_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale
            )
        
        generated_image = result.images[0]
        
        # Resize back to original dimensions if requested
        if resize_to_original and original_size != (512, 512):
            generated_image = generated_image.resize(original_size, Image.LANCZOS)
        
        # Save to bytes buffer
        img_io = io.BytesIO()
        generated_image.save(img_io, format='PNG')
        img_io.seek(0)
        
        print("✅ Generation complete!\n")
        
        return StreamingResponse(
            img_io,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_interior.png"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
