import os
from contextlib import asynccontextmanager
from typing import List, Tuple
import time

import torch
import torch_xla.core.xla_model as xm
# It's good practice to import this for debugging, though not used directly here
import torch_xla.debug.metrics as met
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

# A dictionary to hold the model, a best practice for managing state in FastAPI.
model_handler = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown. The model is loaded and compiled on startup
    to ensure the endpoint is immediately ready for high-performance inference.
    """
    print("🚀 Server starting up...")
    try:
        # 1. Acquire the TPU hardware device
        device = xm.xla_device()
        print("Successfully acquired TPU device.")

        # 2. Load the pre-trained model from Hugging Face
        model_name = 'cross-encoder/ms-marco-MiniLM-L12-v2'
        # Load to CPU first, then move the inner model to the TPU device
        model = CrossEncoder(model_name, device='cpu')
        model.model.to(device)

        # 3. CRITICAL: Set the model to evaluation mode
        model.model.eval()
        print(f"Model '{model_name}' loaded and set to eval mode.")

        # --- 4. NEW: COMPILE THE MODEL WITH THE TORCH_XLA BACKEND ---
        # This is the key change for maximizing performance.
        # It happens only once on startup.
        print("Compiling the model for the XLA backend... (this may take a moment)")
        # We compile the core nn.Module, not the CrossEncoder wrapper
        compiled_model = torch.compile(model.model, backend="openxla")
        model_handler["compiled_model"] = compiled_model
        model_handler["tokenizer"] = model.tokenizer
        print("Model compilation complete.")


        # --- 5. UPDATED: END-TO-END WARM-UP ROUTINE ---
        # Now we warm up the *compiled* model.
        print("Starting end-to-end model warm-up...")
        warmup_batch_sizes = [1, 2, 4, 8]
        dummy_input = [["query", "document"]]

        for batch_size in warmup_batch_sizes:
            print(f"   - Warming up for batch size: {batch_size}...")
            dummy_batch = dummy_input * batch_size
            
            features = model.tokenizer(dummy_batch, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            features = {key: val.to(device) for key, val in features.items()}
            
            # The first call for a new shape/batch size will trigger a re-compile.
            with torch.no_grad():
                outputs = compiled_model(**features)

            # Execution is now synchronous. No xm.mark_step() needed.
            # The .cpu() call simply moves the already-computed result.
            _ = outputs.logits.cpu()

        print("Warm-up complete. Endpoint is fully ready for traffic.")

    except Exception as e:
        print(f"An error occurred during startup or warm-up: {e}")
        model_handler.clear()

    yield  # The application runs here

    print("Server shutting down...")
    model_handler.clear()


# Initialize the FastAPI app with the lifespan manager
app = FastAPI(title="Cross-Encoder Reranker API", lifespan=lifespan)

class RerankRequest(BaseModel):
    instances: List[Tuple[str, str]]

@app.get(os.environ.get('AIP_HEALTH_ROUTE', '/health'), status_code=200)
def health():
    """Health check endpoint required by Vertex AI."""
    return {"status": "healthy" if "compiled_model" in model_handler else "unhealthy"}

@app.post(os.environ.get('AIP_PREDICT_ROUTE', '/predict'))
async def predict(request: RerankRequest):
    """Accepts sentence pairs and returns reranked scores."""
    compiled_model = model_handler.get("compiled_model")
    tokenizer = model_handler.get("tokenizer")
    
    if not compiled_model or not tokenizer:
        return {"error": "Model not loaded or warm-up failed"}, 503

    features = tokenizer(
        request.instances,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    # Get the device from one of the model's parameters
    device = next(compiled_model.parameters()).device
    features = {key: val.to(device) for key, val in features.items()}
    
    with torch.no_grad():
        # Call the compiled model directly
        outputs = compiled_model(**features)

    # --- REMOVED ---
    # xm.mark_step() is no longer needed. The call above is synchronous.
    
    scores = outputs.logits.cpu().tolist()
    
    predictions = [score[0] for score in scores]
    
    return {"predictions": predictions}